from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, ifL

CONSTRAIN_ACTIVE = True

Graph.clear()
Concept.clear()
Relation.clear()

with Graph("temporal_QA_rule") as graph:
    # Group of sentence
    story = Concept(name="story")
    question = Concept(name="question")
    (story_contain,) = story.contains(question)

    answer_class = question(
        name="answer_class",
        ConceptClass=EnumConcept,
        values=["after", "before", "includes", "is_included", "simultaneous", "vague"],
    )

    output_for_loss = question(name="output_for_loss")

    # Inverse Constrains
    inverse = Concept(name="inverse")
    inv_question1, inv_question2 = inverse.has_a(arg1=question, arg2=question)

    # symmetric - if q1 has label, q2 has same label
    inverse_list2 = [(answer_class.simultaneous, answer_class.simultaneous), (answer_class.vague, answer_class.vague)]
    for ans1, ans2 in inverse_list2:
        ifL(
            andL(inverse("s"), ans1(path=("s", inv_question1))),
            ans2(path=("s", inv_question2)),
        )

    ########################################################################################
    ############################ Transitive ##################################################
    transitive = Concept(name="transitive")
    tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

    # A rel B & B rel C -> A rel C
    transitive_1 = [
        answer_class.before,
        answer_class.after,
        answer_class.includes,
        answer_class.is_included,
        answer_class.simultaneous,
        # answer_class.vague,
    ]
    for rel in transitive_1:
        ifL(
            andL(
                transitive("t"),
                rel(path=("t", tran_quest1)),
                rel(path=("t", tran_quest2)),
            ),
            rel(path=("t", tran_quest3)),
        )

    #######################rule 2 #####################################################
    # A rel B & B=C -> A rel C
    for rel in [answer_class.before, answer_class.after, answer_class.includes, answer_class.is_included]:
        ifL(
            andL(
                transitive("t"),
                rel(path=("t", tran_quest1)),
                answer_class.simultaneous(path=("t", tran_quest2)),
            ),
            rel(path=("t", tran_quest3)),
        )

    ############################ before ##################################################
    # # A<B & B includes C => [A<C, A includes C, A vague C]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.before(path=("t", tran_quest1)),
    #         answer_class.includes(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.before(path=("t", tran_quest3)),
    #         answer_class.includes(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    # # A<B & B is_included C -> [before, is_included, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.before(path=("t", tran_quest1)),
    #         answer_class.is_included(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.before(path=("t", tran_quest3)),
    #         answer_class.is_included(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    ########################### after ####################################################
    # # A>B & B includes C => [after, includes, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.after(path=("t", tran_quest1)),
    #         answer_class.includes(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.after(path=("t", tran_quest3)),
    #         answer_class.includes(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    # # A>B & B is_included C -> [after, is_included, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.after(path=("t", tran_quest1)),
    #         answer_class.is_included(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.after(path=("t", tran_quest3)),
    #         answer_class.is_included(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    ######################## is included ##################################################
    # # A is_included B & B<C => [before, is_included, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.is_included(path=("t", tran_quest1)),
    #         answer_class.before(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.before(path=("t", tran_quest3)),
    #         answer_class.is_included(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    # # A is_included B & B>C => [after, is_included, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.is_included(path=("t", tran_quest1)),
    #         answer_class.after(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.after(path=("t", tran_quest3)),
    #         answer_class.is_included(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    ############################ includes ###########################################################
    # # A includes B & B < C -> [before, includes, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.includes(path=("t", tran_quest1)),
    #         answer_class.before(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.before(path=("t", tran_quest3)),
    #         answer_class.includes(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    # # A includes B & B > C -> [after, includes, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.includes(path=("t", tran_quest1)),
    #         answer_class.after(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.after(path=("t", tran_quest3)),
    #         answer_class.includes(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    # # A includes B & B is_included C -> [includes, is_included, simultaneous, vague]
    # ifL(
    #     andL(
    #         transitive("t"),
    #         answer_class.includes(path=("t", tran_quest1)),
    #         answer_class.is_included(path=("t", tran_quest2)),
    #     ),
    #     orL(
    #         answer_class.includes(path=("t", tran_quest3)),
    #         answer_class.is_included(path=("t", tran_quest3)),
    #         answer_class.simultaneous(path=("t", tran_quest3)),
    #         answer_class.vague(path=("t", tran_quest3)),
    #     ),
    # )

    #######################simultaneous#####################################
    # A = B & B rel C -> A rel C
    for rel in [answer_class.before, answer_class.after, answer_class.includes, answer_class.is_included]:
        ifL(
            andL(
                transitive("t"),
                answer_class.simultaneous(path=("t", tran_quest1)),
                rel(path=("t", tran_quest2)),
            ),
            rel(path=("t", tran_quest3)),
        )
