from old_scout import SequenceTemplate


def test_sequence_template_link():
    st = SequenceTemplate()
    st.add_link(0, 2)
    st.add_link(1, 2)
    assert st.get_dependencies(2) == set([0, 1])


def test_sequence_topo_sort():
    st = SequenceTemplate(chord_templates=range(3))
    st.add_link(0, 2)
    st.add_link(0, 1)
    st.add_link(2, 1)
    assert st.eval_order() == [0, 2, 1]
