use std::{cmp::Ordering, collections::{HashMap, HashSet}, iter::zip, sync::Arc};

use itertools::Itertools;
use p3_commit::Pcs;
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use tracing::instrument;
use types::MultiStarkVerifyingKey0;

use crate::{
    air_builders::symbolic::{SymbolicConstraints, SymbolicRapBuilder, get_symbolic_builder, symbolic_expression::SymbolicExpression, symbolic_variable::{Entry, SymbolicVariable}},
    config::{Com, RapPartialProvingKey, StarkGenericConfig, Val},
    interaction::{Interaction, RapPhaseSeq, RapPhaseSeqKind},
    keygen::types::{
        LinearConstraint, MultiStarkProvingKey, ProverOnlySinglePreprocessedData, StarkProvingKey,
        StarkVerifyingKey, TraceWidth, VerifierSinglePreprocessedData,
    },
    rap::AnyRap,
};

pub mod types;
pub mod view;

struct AirKeygenBuilder<SC: StarkGenericConfig> {
    air: Arc<dyn AnyRap<SC>>,
    rap_phase_seq_kind: RapPhaseSeqKind,
    prep_keygen_data: PrepKeygenData<SC>,
}

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
    /// Information for partitioned AIRs.
    partitioned_airs: Vec<AirKeygenBuilder<SC>>,
    max_constraint_degree: usize,
}

impl<'a, SC: StarkGenericConfig> MultiStarkKeygenBuilder<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            partitioned_airs: vec![],
            max_constraint_degree: 0,
        }
    }

    /// The builder will **try** to keep the max constraint degree across all AIRs below this value.
    /// If it is given AIRs that exceed this value, it will still include them.
    ///
    /// Currently this is only used for interaction chunking in FRI logup.
    pub fn set_max_constraint_degree(&mut self, max_constraint_degree: usize) {
        self.max_constraint_degree = max_constraint_degree;
    }

    /// Default way to add a single Interactive AIR.
    /// Returns `air_id`
    #[instrument(level = "debug", skip_all)]
    pub fn add_air(&mut self, air: Arc<dyn AnyRap<SC>>) -> usize {
        self.partitioned_airs.push(AirKeygenBuilder::new(
            self.config.pcs(),
            SC::RapPhaseSeq::ID,
            air,
        ));
        self.partitioned_airs.len() - 1
    }

    /// Consume the builder and generate proving key.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_pk(mut self) -> MultiStarkProvingKey<SC> {
        let air_max_constraint_degree = self
            .partitioned_airs
            .iter()
            .map(|keygen_builder| {
                let max_constraint_degree = keygen_builder.max_constraint_degree();
                tracing::debug!(
                    "{} has constraint degree {}",
                    keygen_builder.air.name(),
                    max_constraint_degree
                );
                max_constraint_degree
            })
            .max()
            .unwrap();
        tracing::info!(
            "Max constraint (excluding logup constraints) degree across all AIRs: {}",
            air_max_constraint_degree
        );
        if self.max_constraint_degree != 0 && air_max_constraint_degree > self.max_constraint_degree
        {
            // This means the quotient polynomial is already going to be higher degree, so we
            // might as well use it.
            tracing::info!(
                "Setting max_constraint_degree from {} to {air_max_constraint_degree}",
                self.max_constraint_degree
            );
            self.max_constraint_degree = air_max_constraint_degree;
        }
        // First pass: get symbolic constraints and interactions but RAP phase constraints are not
        // final
        let symbolic_constraints_per_air = self
            .partitioned_airs
            .iter()
            .map(|keygen_builder| keygen_builder.get_symbolic_builder(None).constraints())
            .collect_vec();
        // Note: due to the need to go through a trait, there is some duplicate computation
        // (e.g., FRI logup will calculate the interaction chunking both here and in the second pass
        // below)
        let rap_partial_pk_per_air = self
            .config
            .rap_phase_seq()
            .generate_pk_per_air(&symbolic_constraints_per_air, self.max_constraint_degree);
        let pk_per_air: Vec<_> = zip(self.partitioned_airs, rap_partial_pk_per_air)
            .map(|(keygen_builder, rap_partial_pk)| {
                // Second pass: get final constraints, where RAP phase constraints may have changed
                keygen_builder.generate_pk(rap_partial_pk, self.max_constraint_degree)
            })
            .collect();

        for pk in pk_per_air.iter() {
            let width = &pk.vk.params.width;
            tracing::info!("{:<20} | Quotient Deg = {:<2} | Prep Cols = {:<2} | Main Cols = {:<8} | Perm Cols = {:<4} | {:4} Constraints | {:3} Interactions",
                pk.air_name,
                pk.vk.quotient_degree,
                width.preprocessed.unwrap_or(0),
                format!("{:?}",width.main_widths()),
                format!("{:?}",width.after_challenge.iter().map(|&x| x * <SC::Challenge as FieldExtensionAlgebra<Val<SC>>>::D).collect_vec()),
                pk.vk.symbolic_constraints.constraints.constraint_idx.len(),
                pk.vk.symbolic_constraints.interactions.len(),
            );
            tracing::debug!(
                "On Buses {:?}",
                pk.vk
                    .symbolic_constraints
                    .interactions
                    .iter()
                    .map(|i| i.bus_index)
                    .collect_vec()
            );
            #[cfg(feature = "metrics")]
            {
                let labels = [("air_name", pk.air_name.clone())];
                metrics::counter!("quotient_deg", &labels).absolute(pk.vk.quotient_degree as u64);
                // column info will be logged by prover later
                metrics::counter!("constraints", &labels)
                    .absolute(pk.vk.symbolic_constraints.constraints.constraint_idx.len() as u64);
                metrics::counter!("interactions", &labels)
                    .absolute(pk.vk.symbolic_constraints.interactions.len() as u64);
            }
        }

        let num_airs = symbolic_constraints_per_air.len();
        let base_order = Val::<SC>::order().to_u32_digits()[0];
        let mut count_weight_per_air_per_bus_index = HashMap::new();

        // We compute the a_i's for the constraints of the form a_0 n_0 + ... + a_{k-1} n_{k-1} <
        // a_k, First the constraints that the total number of interactions on each bus is
        // at most the base field order.
        for (i, constraints_per_air) in symbolic_constraints_per_air.iter().enumerate() {
            for interaction in &constraints_per_air.interactions {
                // Also make sure that this of interaction is valid given the security params.
                // +1 because of the bus
                let max_msg_len = self
                    .config
                    .rap_phase_seq()
                    .log_up_security_params()
                    .max_message_length();
                // plus one because of the bus
                let total_message_length = interaction.message.len() + 1;
                assert!(
                    total_message_length <= max_msg_len,
                    "interaction message with bus has length {}, which is more than max {max_msg_len}",
                    total_message_length,
                );

                let b = interaction.bus_index;
                let constraint = count_weight_per_air_per_bus_index
                    .entry(b)
                    .or_insert_with(|| LinearConstraint {
                        coefficients: vec![0; num_airs],
                        threshold: base_order,
                    });
                constraint.coefficients[i] += interaction.count_weight;
            }
        }

        // Sorting by bus index is not necessary, but makes debugging/testing easier.
        let mut trace_height_constraints = count_weight_per_air_per_bus_index
            .into_iter()
            .sorted_by_key(|(bus_index, _)| *bus_index)
            .map(|(_, constraint)| constraint)
            .collect_vec();

        let log_up_security_params = self.config.rap_phase_seq().log_up_security_params();

        // Add a constraint for the total number of interactions.
        trace_height_constraints.push(LinearConstraint {
            coefficients: symbolic_constraints_per_air
                .iter()
                .map(|c| c.interactions.len() as u32)
                .collect(),
            threshold: log_up_security_params.max_interaction_count,
        });

        let pre_vk: MultiStarkVerifyingKey0<SC> = MultiStarkVerifyingKey0 {
            per_air: pk_per_air.iter().map(|pk| pk.vk.clone()).collect(),
            trace_height_constraints: trace_height_constraints.clone(),
            log_up_pow_bits: log_up_security_params.log_up_pow_bits,
        };
        // To protect against weak Fiat-Shamir, we hash the "pre"-verifying key and include it in
        // the final verifying key. This just needs to commit to the verifying key and does
        // not need to be verified by the verifier, so we just use bincode to serialize it.
        let vk_bytes = bitcode::serialize(&pre_vk).unwrap();
        tracing::info!("pre-vkey: {} bytes", vk_bytes.len());
        // Purely to get type compatibility and convenience, we hash using pcs.commit as a single
        // row
        let vk_as_row = RowMajorMatrix::new_row(
            vk_bytes
                .into_iter()
                .map(Val::<SC>::from_canonical_u8)
                .collect(),
        );
        let pcs = self.config.pcs();
        let deg_1_domain = pcs.natural_domain_for_degree(1);
        let (vk_pre_hash, _) = pcs.commit(vec![(deg_1_domain, vk_as_row)]);

        MultiStarkProvingKey {
            per_air: pk_per_air,
            trace_height_constraints,
            max_constraint_degree: self.max_constraint_degree,
            log_up_pow_bits: log_up_security_params.log_up_pow_bits,
            vk_pre_hash,
        }
    }
}

impl<SC: StarkGenericConfig> AirKeygenBuilder<SC> {
    fn new(pcs: &SC::Pcs, rap_phase_seq_kind: RapPhaseSeqKind, air: Arc<dyn AnyRap<SC>>) -> Self {
        let prep_keygen_data = compute_prep_data_for_air(pcs, air.as_ref());
        AirKeygenBuilder {
            air,
            rap_phase_seq_kind,
            prep_keygen_data,
        }
    }

    fn max_constraint_degree(&self) -> usize {
        self.get_symbolic_builder(None)
            .constraints()
            .max_constraint_degree()
    }

    fn generate_pk(
        self,
        rap_partial_pk: RapPartialProvingKey<SC>,
        max_constraint_degree: usize,
    ) -> StarkProvingKey<SC> {
        let air_name = self.air.name();

        let symbolic_builder = self.get_symbolic_builder(Some(max_constraint_degree));
        let params = symbolic_builder.params();
        let symbolic_constraints = symbolic_builder.constraints();

        //-----------------------------------------
        println!("-----Constraints for {air_name}-----");
        println!("-----Used Columns-------------------");
        println!("{}", placeholder_column_names(&symbolic_constraints));
        println!("-----Extracted constraints----------");
        for (idx, constraint) in symbolic_constraints.constraints.iter().enumerate() {
            let constraint_text = format!(
                "  @[simp]\n  def constraint_{idx} {{C : Type → Type → Type}} {{F ExtF : Type}} [Field F] [Field ExtF] [Circuit F ExtF C] (c : C F ExtF) (row: ℕ) :=\n    {} = 0\n",
                symbolic_expression_to_string(constraint, "", None)
            );
            if constraint_text.contains("Circuit.permutation") {
                let commented = constraint_text
                    .split("\n")
                    .map(|line| format!("-- {line}"))
                    .join("\n");
                println!("{commented}");
            } else {
                println!("{constraint_text}");
            }
        }
        println!("  def constrain_interactions {{C : Type → Type → Type}} {{F ExtF : Type}} [Field F] [Field ExtF] [Circuit F ExtF C] (c : C F ExtF) :=");
        println!("    Circuit.buses c = λ index =>");

        let mut interactions_by_bus: HashMap<u16, Vec<Interaction<_>>> = HashMap::new();
        for interaction in symbolic_constraints.interactions.iter() {
            if let Some(list) = interactions_by_bus.get_mut(&interaction.bus_index) {
                list.push(interaction.clone());
            } else {
                interactions_by_bus.insert(interaction.bus_index, vec![interaction.clone()]);
            }
        }

        for (idx, (bus_idx, interactions)) in interactions_by_bus
            .iter()
            .sorted_by(|(a, _), (c, _)| (**a).cmp(*c))
            .enumerate()
        {
            let all_rows = "(List.range (Circuit.last_row c + 1))";
            let row_interactions = interactions
                .iter()
                .map(|interaction| {
                    let multiplicity = symbolic_expression_to_string(
                        &interaction.count,
                        "",
                        None
                    );
                    let data = format!(
                        "[{}]",
                        interaction
                            .message
                            .iter()
                            .map(|x| symbolic_expression_to_string(x, "", None))
                            .join(", ")
                    );
                    format!("({multiplicity}, {data})")
                })
                .join(", ");
            let expr = format!(
                "{all_rows}.flatMap (λ row => [{row_interactions}])"
            );

            println!(
                "      {}if index = {} then {expr}",
                if idx == 0 { "" } else { "else " },
                bus_idx,
            )
            // println!("\nInteraction {idx}:");
            // println!("    Bus {}", interaction.bus_index);
            // println!(
            //     "    Message: ({})",
            //     interaction
            //         .message
            //         .iter()
            //         .map(|x| symbolic_expression_to_string(x, "", false, None))
            //         .join(", ")
            // );
            // println!(
            //     "    Count: {}",
            //     symbolic_expression_to_string(&interaction.count, "", false, None)
            // );
            // println!(
            //     "    Count weight: {}",
            //     interaction.count_weight
            // );
        }
        if interactions_by_bus.is_empty() {
            println!("    []")
        } else {
            println!("    else []")
        }
        println!("-----Constraint simplification------");
        let simplification_proof = [
            "apply Iff.intro",
            ". intro h",
            "  simp [openvm_encapsulation, NAME_constraint_and_interaction_simplification] at h",
            "  simp only [NAME_constraint_and_interaction_simplification]",
            "  exact h",
            ". intro h",
            "  simp [openvm_encapsulation, NAME_constraint_and_interaction_simplification]",
            "  simp only [NAME_constraint_and_interaction_simplification] at h",
            "  exact h",
        ].join("\n");
        for (idx, constraint) in symbolic_constraints.constraints.iter().enumerate() {
            let constraint_text = format!(
                "{}",
                symbolic_expression_to_string(constraint, "", None)
            );

            let simplified_constraint_text = [
                format!("@[NAME_constraint_and_interaction_simplification]"),
                format!("def constraint_{idx} (air : Valid_NAME F ExtF) (row : ℕ) : Prop :="),
                format!("  sorry")
            ].join("\n");

            let simplified_of_extracted = [
                format!("@[NAME_air_simplification]"),
                format!("lemma constraint_{idx}_of_extraction"),
                format!("    (air : Valid_NAME F ExtF) (row : ℕ)"),
                format!(": NAME.extraction.constraint_{idx} air row ↔ constraint_{idx} air row := by"),
                simplification_proof.clone()
            ].join("\n");

            let output_text = format!(
                "{simplified_constraint_text}\n\n{simplified_of_extracted}"
            );

            if constraint_text.contains("Circuit.permutation") {
                let commented = output_text
                    .split("\n")
                    .map(|line| format!("-- {line}"))
                    .join("\n");
                println!("{commented}\n");
            } else {
                println!("{output_text}\n");
            }
        }
        println!("-----Interaction simplification-----");
        let mut full_expr = vec![];
        for (bus_idx, interactions) in interactions_by_bus
            .iter()
            .sorted_by(|(a, _), (c, _)| (**a).cmp(*c))
        {
            let (bus_name, bus_idx_name) = if *bus_idx == 0 {
                (format!("execution"), format!("ExecutionBus"))
            } else if *bus_idx == 1 {
                (format!("memory"), format!("MemoryBus"))
            } else if *bus_idx == 4 {
                (format!("rangeChecker"), format!("RangeCheckerBus"))
            } else if *bus_idx == 8 {
                (format!("readInstruction"), format!("ReadInstructionBus"))
            } else if *bus_idx == 9 {
                (format!("bitwise"), format!("BitwiseBus"))
            } else if *bus_idx == 11 {
                (format!("rangeTupleChecker"), format!("RangeTupleCheckerBus"))
            } else {
                (format!("bus_{bus_idx}"), format!("{bus_idx}"))
            };

            let row_expression = [
                format!("@[NAME_constraint_and_interaction_simplification]",),
                format!("def {bus_name}Bus_row (air : Valid_NAME F ExtF) (row : ℕ) : List (F × List F) :=",),
                format!("  sorry")
            ].join("\n");

            let constrain_lemma = [
                format!("lemma constrain_{bus_name}_interactions"),
                format!("  (air : Valid_NAME F ExtF)"),
                format!("  (h : NAME.extraction.constrain_interactions air)"),
                format!(":"),
                format!("  air.buses {bus_idx_name} = (List.range (air.last_row + 1)).flatMap (λ row => {bus_name}Bus_row air row)"),
                format!(":= by"),
                format!("  unfold NAME.extraction.constrain_interactions at h"),
                format!("  simp [openvm_encapsulation] at h"),
                format!("  simp [h]; clear h"),
                format!("  rfl"),
            ].join("\n");

            println!("{row_expression}\n\n{constrain_lemma}\n\n");

            full_expr.push(format!(
                "if index = {bus_idx_name} then (List.range (air.last_row + 1)).flatMap ({bus_name}Bus_row air)"
            ))
        }

        full_expr.push(format!("[]"));

        let full_expr = full_expr.join("\nelse ");

        let constrain_interactions_lemma = [
            format!("def constrain_interactions (air : Valid_NAME F ExtF) : Prop :="),
            format!("air.buses = fun index ↦"),
            full_expr
        ].join("\n");

        let constrain_interactions_of_extraction_lemma = [
            "@[NAME_air_simplification]",
            "lemma constrain_interactions_of_extraction",
            "  (air : Valid_NAME F ExtF)",
            "  (h : NAME.extraction.constrain_interactions air)",
            ": constrain_interactions air := by",
            "  unfold NAME.extraction.constrain_interactions at h",
            "  simp [openvm_encapsulation] at h",
            "  exact h",
        ].join("\n");

        println!("{constrain_interactions_lemma}\n\n{constrain_interactions_of_extraction_lemma}\n\n");

        println!("-----All hold definitions-----------");

        let num_constraints = symbolic_constraints.constraints.len();

        let num_comment_constraints = symbolic_constraints
            .constraints
            .iter()
            .filter(|constraint| {
                let constraint_text = format!(
                    "{}",
                    symbolic_expression_to_string(constraint, "", None)
                );
                constraint_text.contains("Circuit.permutation")
            })
            .count();

        let extracted_row_constraint_list = (0..num_constraints)
            .map(|idx| {
                let constraint = format!("    NAME.extraction.constraint_{idx} air row,");

                if idx >= num_constraints - num_comment_constraints {
                    format!("-- {constraint}")
                } else {
                    constraint
                }
            })
            .join("\n");

        let extract_row_constraint_list_def = [
            format!("@[simp]"),
            format!("def extracted_row_constraint_list"),
            format!("  [Field ExtF]"),
            format!("  (air : Valid_NAME FBB ExtF)"),
            format!("  (row : ℕ)"),
            format!(": List Prop :="),
            format!("  ["),
            extracted_row_constraint_list,
            format!("  ]"),
        ].join("\n");

        let all_hold_def = [
            "@[simp]",
            "def allHold",
            "  [Field ExtF]",
            "  (air : Valid_NAME FBB ExtF)",
            "  (row : ℕ)",
            "  (_ : row ≤ air.last_row)",
            ": Prop :=",
            "  NAME.extraction.constrain_interactions air ∧",
            "  List.Forall (·) (extracted_row_constraint_list air row)",
        ].join("\n");

        let row_constraint_list = (0..num_constraints)
            .map(|idx| {
                let constraint = format!("    constraint_{idx} air row,");

                if idx >= num_constraints - num_comment_constraints {
                    format!("-- {constraint}")
                } else {
                    constraint
                }
            })
            .join("\n");

        let row_constraint_list_def = [
            format!("@[simp]"),
            format!("def row_constraint_list"),
            format!("  [Field ExtF]"),
            format!("  (air : Valid_NAME FBB ExtF)"),
            format!("  (row : ℕ)"),
            format!(": List Prop :="),
            format!("  ["),
            row_constraint_list,
            format!("  ]"),
        ].join("\n");

        let all_hold_simplified = [
            "@[simp]",
            "def allHold_simplified",
            "  [Field ExtF]",
            "  (air : Valid_NAME FBB ExtF)",
            "  (row : ℕ)",
            "  (_ : row ≤ air.last_row)",
            ": Prop :=",
            "  constrain_interactions air ∧",
            "  List.Forall (·) (row_constraint_list air row)",
        ].join("\n");

        let all_hold_simplified_of_all_hold = [
            "lemma allHold_simplified_of_allHold",
            "  [Field ExtF]",
            "  (air : Valid_NAME FBB ExtF)",
            "  (row : ℕ)",
            "  (h_row : row ≤ air.last_row)",
            ": allHold air row h_row ↔ allHold_simplified air row h_row := by",
            "  unfold allHold allHold_simplified",
            "  apply Iff.and",
            "  . unfold NAME.extraction.constrain_interactions",
            "    simp [openvm_encapsulation]",
            "    rfl",
            "  . simp only [extracted_row_constraint_list,",
            "              row_constraint_list,",
            "              NAME_air_simplification]",
        ].join("\n");

        let all_hold_section = [
            extract_row_constraint_list_def,
            all_hold_def,
            row_constraint_list_def,
            all_hold_simplified,
            all_hold_simplified_of_all_hold
        ].join("\n\n");

        println!("{all_hold_section}");

        println!("------");














        //-----------------------------------------






        let log_quotient_degree = symbolic_constraints.get_log_quotient_degree();
        let quotient_degree = 1 << log_quotient_degree;

        let Self {
            prep_keygen_data:
                PrepKeygenData {
                    verifier_data: prep_verifier_data,
                    prover_data: prep_prover_data,
                },
            ..
        } = self;

        let vk: StarkVerifyingKey<Val<SC>, Com<SC>> = StarkVerifyingKey {
            preprocessed_data: prep_verifier_data,
            params,
            symbolic_constraints: symbolic_constraints.into(),
            quotient_degree,
            rap_phase_seq_kind: self.rap_phase_seq_kind,
        };
        StarkProvingKey {
            air_name,
            vk,
            preprocessed_data: prep_prover_data,
            rap_partial_pk,
        }
    }

    fn get_symbolic_builder(
        &self,
        max_constraint_degree: Option<usize>,
    ) -> SymbolicRapBuilder<Val<SC>> {
        let width = TraceWidth {
            preprocessed: self.prep_keygen_data.width(),
            cached_mains: self.air.cached_main_widths(),
            common_main: self.air.common_main_width(),
            after_challenge: vec![],
        };
        get_symbolic_builder(
            self.air.as_ref(),
            &width,
            &[],
            &[],
            SC::RapPhaseSeq::ID,
            max_constraint_degree.unwrap_or(0),
        )
    }
}

pub(super) struct PrepKeygenData<SC: StarkGenericConfig> {
    pub verifier_data: Option<VerifierSinglePreprocessedData<Com<SC>>>,
    pub prover_data: Option<ProverOnlySinglePreprocessedData<SC>>,
}

impl<SC: StarkGenericConfig> PrepKeygenData<SC> {
    pub fn width(&self) -> Option<usize> {
        self.prover_data.as_ref().map(|d| d.trace.width())
    }
}

fn compute_prep_data_for_air<SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    air: &dyn AnyRap<SC>,
) -> PrepKeygenData<SC> {
    let preprocessed_trace = air.preprocessed_trace();
    let vpdata_opt = preprocessed_trace.map(|trace| {
        let domain = pcs.natural_domain_for_degree(trace.height());
        let (commit, data) = pcs.commit(vec![(domain, trace.clone())]);
        let vdata = VerifierSinglePreprocessedData { commit };
        let pdata = ProverOnlySinglePreprocessedData {
            trace: Arc::new(trace),
            data: Arc::new(data),
        };
        (vdata, pdata)
    });
    if let Some((vdata, pdata)) = vpdata_opt {
        PrepKeygenData {
            prover_data: Some(pdata),
            verifier_data: Some(vdata),
        }
    } else {
        PrepKeygenData {
            prover_data: None,
            verifier_data: None,
        }
    }
}





//------------------------------------------------------------------------------------------------------------------------------------------------
fn collect_variables<F>(expression: &SymbolicExpression<F>, leaves: &mut HashSet<SymbolicVariable<F>>) -> ()
    where F: Clone + std::cmp::Eq + std::hash::Hash
{
    match expression {
        SymbolicExpression::Variable(symbolic_variable) => {
            leaves.insert(symbolic_variable.clone());
        },
        SymbolicExpression::Add { x, y, degree_multiple: _ } => {
            collect_variables(x, leaves);
            collect_variables(y, leaves);
        },
        SymbolicExpression::Sub { x, y, degree_multiple: _ } => {
            collect_variables(x, leaves);
            collect_variables(y, leaves);
        },
        SymbolicExpression::Neg { x, degree_multiple: _ } => {
            collect_variables(x, leaves);
        },
        SymbolicExpression::Mul { x, y, degree_multiple: _ } => {
            collect_variables(x, leaves);
            collect_variables(y, leaves);
        },
        _ => {}
    }
}


fn get_entry_type_id(entry: &Entry) -> u8 {
    match entry {
        Entry::Preprocessed { offset: _ } => 0,
        Entry::Main { part_index: _, offset: _ } => 1,
        Entry::Permutation { offset: _ } => 2,
        Entry::Public => 3,
        Entry::Challenge => 4,
        Entry::Exposed => 5,
    }
}

fn placeholder_column_names<F>(constraints: &SymbolicConstraints<F>) -> String
    where F: Clone + std::cmp::Eq + std::hash::Hash
{
    let leaves = {
        let mut leaves = HashSet::new();

        constraints.constraints.iter().for_each(|expr| {
            collect_variables(expr, &mut leaves);
        });

        constraints.interactions.iter().for_each(|interaction| {
            collect_variables(&interaction.count, &mut leaves);
            interaction.message.iter().for_each(|expr| {
                collect_variables(expr, &mut leaves);
            });
        });

        let leaves = leaves.into_iter().sorted_by(|lhs, rhs| {
            let type_order = get_entry_type_id(&lhs.entry).cmp(&get_entry_type_id(&rhs.entry));

            let index_order = lhs.index.cmp(&rhs.index);

            let (part_index_order, offset_order) = match (lhs.entry, rhs.entry) {
                (Entry::Preprocessed { offset: l_offset }, Entry::Preprocessed { offset: r_offset }) => (Ordering::Equal, l_offset.cmp(&r_offset)),
                (Entry::Main { part_index: l_part_index, offset: l_offset }, Entry::Main { part_index: r_part_index, offset: r_offset }) => (l_part_index.cmp(&&r_part_index), l_offset.cmp(&r_offset)),
                (Entry::Permutation { offset: l_offset }, Entry::Permutation { offset: r_offset }) => (Ordering::Equal, l_offset.cmp(&r_offset)),
                _ => (Ordering::Equal, Ordering::Equal),
            };

            type_order.then(part_index_order).then(index_order).then(offset_order)
        }).collect_vec();

        leaves
    };

    leaves.iter().map(|leaf| {
        let column = leaf.index;
        match leaf.entry {
            Entry::Preprocessed { offset } =>
                format!("--def Circuit._ (c: Circuit F ExtF) (row: N) := c.preprocessed (column := {column}) (row := row) (rotation := {offset})"),
            Entry::Main { part_index, offset } =>
                format!("--def Circuit._ (c: Circuit F ExtF) (row: N) := c.main (id := {part_index}) (column := {column}) (row := row) (rotation := {offset})"),
            Entry::Permutation { offset } =>
                format!("--def Circuit._ (c: Circuit F ExtF) (row: N) := c.permutation (column := {column}) (row := row) (rotation := {offset})"),
            Entry::Public =>
                format!("--def Circuit._ (c: Circuit F ExtF) := c.public (index := {column})"),
            Entry::Challenge =>
                format!("--def Circuit._ (c: Circuit F ExtF) := c.challenge (index := {column})"),
            Entry::Exposed =>
                format!("--def Circuit._ (c: Circuit F ExtF) := c.exposed (index := {column})"),
        }
    }).join("\n")
}

fn symbolic_expression_to_string<F: Field>(x: &SymbolicExpression<F>, scoping: &str, characteristic: Option<u32>) -> String {
    let x = x.clone();
    symbolic_expression_to_string_impl(&x, scoping, characteristic)
}

fn symbolic_expression_to_string_impl<F: Field>(x: &SymbolicExpression<F>, scoping: &str, characteristic: Option<u32>) -> String {
    match x {
        SymbolicExpression::Variable(symbolic_variable) =>
            format!(
                "{scoping}{}",
                match symbolic_variable.entry {
                    Entry::Preprocessed{offset}=>format!("(Circuit.preprocessed c (column := {}) (row := row) (rotation := {offset}))",symbolic_variable.index),
                    Entry::Main{offset, part_index}=>format!("(Circuit.main c (id := {part_index}) (column := {}) (row := row) (rotation := {offset}))",symbolic_variable.index),
                    Entry::Permutation{offset}=>format!("(Circuit.permutation c (column := {}) (row := row) (rotation := {offset}))",symbolic_variable.index),
                    Entry::Public=>format!("(Circuit.public c (index := {}))",symbolic_variable.index),
                    Entry::Challenge=>format!("(Circuit.challenge c (index := {}))",symbolic_variable.index),
                    Entry::Exposed =>format!("(Circuitc.exposed c (index := {}))",symbolic_variable.index),
                },
                
            ),
        SymbolicExpression::IsFirstRow => format!("(Circuit.isFirstRow c row)"),
        SymbolicExpression::IsLastRow => format!("(Circuit.isLastRow c row)"),
        SymbolicExpression::IsTransition => format!("(Circuit.isTransitionRow c row)"),
        SymbolicExpression::Constant(x) => {
            let num = str::parse::<u32>(&format!("{x}"));
            match num {
                Ok(num) => {
                    match characteristic {
                        Some(characteristic) => {
                            if num >= characteristic {
                                format!("{x}")
                            } else if characteristic - num < num {
                                format!("-{}", characteristic - num)
                            } else {
                                format!("{x}")
                            }
                        },
                        None => format!("{x}"),
                    }
                },
                Err(_) => format!("{x}"),
            }
        },
        SymbolicExpression::Add { x, y, degree_multiple } => {
            let lhs = symbolic_expression_to_string_impl(&x, scoping, characteristic);
            let rhs = symbolic_expression_to_string_impl(&y, scoping, characteristic);
            format!("({lhs} + {rhs})")
        },
        SymbolicExpression::Sub { x, y, degree_multiple } => {
            let lhs = symbolic_expression_to_string_impl(&x, scoping, characteristic);
            let rhs = symbolic_expression_to_string_impl(&y, scoping, characteristic);
            format!("({lhs} - {rhs})")
        },
        SymbolicExpression::Neg { x, degree_multiple } => {
            let leaf = symbolic_expression_to_string_impl(&x, scoping, characteristic);
            format!("-({leaf})")
        },
        SymbolicExpression::Mul { x, y, degree_multiple } => {
            let lhs = symbolic_expression_to_string_impl(&x, scoping, characteristic);
            let rhs = symbolic_expression_to_string_impl(&y, scoping, characteristic);
            format!("({lhs} * {rhs})")
        },
    }
}
