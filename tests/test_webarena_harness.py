from jevany.webarena_harness import (
    bounded_observation,
    executable_choice,
    extract_goal,
    navigation_choices,
    validate_browser_action,
)


OBS = """[12] link 'Orders'
[13] textbox 'Search'
[14] button 'Submit'
[15] menuitem 'Office Products' hasPopup: menu
URL: http://example.test
OBJECTIVE: Find the most recent pending order
PREVIOUS ACTION: None"""


def test_extract_goal_and_keep_frame_tail():
    assert extract_goal(OBS) == "Find the most recent pending order"
    clipped = bounded_observation("x" * 15000 + "\nURL: u\nOBJECTIVE: g", 100)
    assert "URL: u" in clipped


def test_navigation_choices_excludes_open_text_fields():
    choices = navigation_choices(OBS)
    assert any("click [12]" in value for value in choices.values())
    assert not any("hover [12]" in value for value in choices.values())
    assert any("hover [15]" in value for value in choices.values())
    assert not any("click [15]" in value for value in choices.values())
    assert any("click [14]" in value for value in choices.values())
    assert not any("[13]" in value for value in choices.values())
    assert executable_choice("click [12] — link 'Orders'") == "click [12]"


def test_browser_action_is_grounded_and_strict():
    assert validate_browser_action("click [12]", OBS) == "click [12]"
    assert validate_browser_action("type [13] [pending] [1]", OBS).startswith("type")
    try:
        validate_browser_action("click [999]", OBS)
    except ValueError as error:
        assert "absent" in str(error)
    else:
        raise AssertionError("ungrounded element id should fail")
