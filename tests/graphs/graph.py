from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import andL, ifL, orL


def get_graph(
    symmetric: bool = False,
    inverse_relation: bool = False,
    transitive_determin: bool = False,
    transitive_non_determin: bool = False,
):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    transitive = None
    inverse = None
    tran_quest1 = None
    tran_quest2 = None
    tran_quest3 = None
    inv_quest1 = None
    inv_quest2 = None

    with Graph("temporal_QA_rule") as graph:
        story = Concept(name="story")
        question = Concept(name="question")
        (story_contain,) = story.contains(question)

        answer_class = question(
            name="answer_class",
            ConceptClass=EnumConcept,
            values=["after", "before", "includes", "is_included", "simultaneous", "vague"],
        )

        if symmetric:
            inverse = Concept(name="inverse")
            inv_quest1, inv_quest2 = inverse.has_a(arg1=question, arg2=question)

            # symmetric - if q1 has label, q2 has same label
            inverse_list2 = [
                (answer_class.simultaneous, answer_class.simultaneous),
                (answer_class.vague, answer_class.vague),
            ]
            for ans1, ans2 in inverse_list2:
                ifL(
                    andL(inverse("s"), ans1(path=("s", inv_quest1))),
                    ans2(path=("s", inv_quest2)),
                )

        if inverse_relation:
            inverse = Concept(name="inverse")
            inv_quest1, inv_quest2 = inverse.has_a(arg1=question, arg2=question)

            # inverse - if q1 has label, q2 has inverse label
            inverse_list = [
                (answer_class.before, answer_class.after),
                (answer_class.after, answer_class.before),
                (answer_class.includes, answer_class.is_included),
                (answer_class.is_included, answer_class.includes),
            ]
            for ans1, ans2 in inverse_list:
                ifL(
                    andL(inverse("s"), ans1(path=("s", inv_quest1))),
                    ans2(path=("s", inv_quest2)),
                )

        if transitive_determin:
            transitive = Concept(name="transitive")
            tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

            # A rel B & B rel C -> A rel C
            transitive_1 = [
                answer_class.before,
                answer_class.after,
                answer_class.includes,
                answer_class.is_included,
                answer_class.simultaneous,
                answer_class.vague,
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

        if transitive_non_determin:
            transitive = Concept(name="transitive")
            tran_quest1, tran_quest2, tran_quest3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

            # A<B & B includes C => [A<C, A includes C, A vague C]
            ifL(
                andL(
                    transitive("t"),
                    answer_class.before(path=("t", tran_quest1)),
                    answer_class.includes(path=("t", tran_quest2)),
                ),
                orL(
                    answer_class.before(path=("t", tran_quest3)),
                    answer_class.includes(path=("t", tran_quest3)),
                    answer_class.vague(path=("t", tran_quest3)),
                ),
            )

    return (
        graph,
        story,
        question,
        transitive,
        inverse,
        answer_class,
        story_contain,
        tran_quest1,
        tran_quest2,
        tran_quest3,
        inv_quest1,
        inv_quest2,
    )
