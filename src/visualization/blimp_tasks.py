# --- BLiMP Task List (Normalized) ---
BLIMP_TASKS = [
    'adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement',
    'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_np_island',
    'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction',
    'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1',
    'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
    'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2',
    'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun',
    'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2',
    'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2',
    'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive',
    'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
    'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2',
    'left_branch_island_echo_question', 'left_branch_island_simple_question',
    'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2',
    'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2',
    'principle_a_c_command', 'principle_a_case_1', 'principle_a_case_2',
    'principle_a_domain_1', 'principle_a_domain_2', 'principle_a_domain_3',
    'principle_a_reconstruction', 'regular_plural_subject_verb_agreement_1',
    'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present',
    'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1',
    'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2',
    'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap',
    'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance',
    'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance'
]

# --- Task Groups from BLiMP Paper ---
BLIMP_GROUPS = {
    "Anaphor Agreement": [
        "anaphor_gender_agreement", "anaphor_number_agreement"
    ],
    "Argument Structure": [
        "animate_subject_passive", "animate_subject_trans", "causative",
        "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2", "transitive"
    ],
    "Binding": [
        "principle_a_c_command", "principle_a_case_1", "principle_a_case_2",
        "principle_a_domain_1", "principle_a_domain_2", "principle_a_domain_3",
        "principle_a_reconstruction"
    ],
    "Control/Raising": [
        "tough_vs_raising_1", "tough_vs_raising_2", "existential_there_subject_raising", "existential_there_object_raising", "expletive_it_object_raising"
    ],
    "Determiner-Noun Agreement": [
        "determiner_noun_agreement_1", "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1"
    ],
    "Ellipsis": [
        "ellipsis_n_bar_1", "ellipsis_n_bar_2"
    ],
    "Filler-Gap Dependencies": [
        "wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance"
    ],
    "Irregular Forms": [
        "irregular_past_participle_adjectives", "irregular_past_participle_verbs"
    ],
    "Island Effects": [
        "adjunct_island", "complex_np_island", "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction", "left_branch_island_echo_question",
        "left_branch_island_simple_question", "sentential_subject_island", "wh_island"
    ],
    "NPI Licensing": [
        "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
        "only_npi_licensor_present", "only_npi_scope", "sentential_negation_npi_licensor_present",
        "sentential_negation_npi_scope"
    ],
    "Quantifiers": [
        "existential_there_quantifiers_1", "existential_there_quantifiers_2",
        "superlative_quantifiers_1", "superlative_quantifiers_2"
    ],
    "Subject-Verb Agreement": [
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2", "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2", "distractor_agreement_relational_noun", "distractor_agreement_relative_clause"
    ]
}


def get_task_group(task_name):
    """
    Get the group name for a given task.
    
    Args:
        task_name: Name of the BLiMP task
        
    Returns:
        Group name or None if task not found
    """
    for group, tasks in BLIMP_GROUPS.items():
        if task_name in tasks:
            return group
    return None
