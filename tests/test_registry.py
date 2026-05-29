import pytest

from lcmvrsi.utils.registry import Registry


def test_register_get_and_names():
    reg: Registry[object] = Registry("widget")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A
    assert reg.names() == ["a"]


def test_duplicate_registration_raises():
    reg: Registry[object] = Registry("widget")

    @reg.register("a")
    class A:
        pass

    with pytest.raises(ValueError):

        @reg.register("a")
        class B:
            pass


def test_unknown_name_raises_keyerror():
    reg: Registry[object] = Registry("widget")
    with pytest.raises(KeyError):
        reg.get("missing")
