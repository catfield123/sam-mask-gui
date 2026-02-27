"""Tests for src.utils.check_packages."""

import pytest


class TestCheckSam2Installed:
    """Tests for check_sam2_installed."""

    def test_sam2_contract(self) -> None:
        """Returns (bool, str | None); if not installed, message mentions sam2 or error."""
        from src.utils.check_packages import check_sam2_installed

        installed, msg = check_sam2_installed()
        assert isinstance(installed, bool)
        assert msg is None or isinstance(msg, str)
        if not installed:
            assert "sam2" in (msg or "").lower() or "not found" in (msg or "").lower() or "error" in (msg or "").lower()

    def test_check_sam2_returns_tuple(self) -> None:
        """check_sam2_installed returns a 2-tuple."""
        from src.utils.check_packages import check_sam2_installed

        result = check_sam2_installed()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestCheckSam3Installed:
    """Tests for check_sam3_installed."""

    def test_check_sam3_returns_tuple(self) -> None:
        """check_sam3_installed returns a 2-tuple."""
        from src.utils.check_packages import check_sam3_installed

        result = check_sam3_installed()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert result[1] is None or isinstance(result[1], str)


class TestCheckAllPackages:
    """Tests for check_all_packages."""

    def test_returns_dict_with_sam2_and_sam3_keys(self) -> None:
        """check_all_packages returns dict with 'sam2' and 'sam3' keys."""
        from src.utils.check_packages import check_all_packages

        result = check_all_packages()
        assert isinstance(result, dict)
        assert "sam2" in result
        assert "sam3" in result

    def test_each_value_is_tuple_bool_optional_str(self) -> None:
        """Each value is (bool, str | None)."""
        from src.utils.check_packages import check_all_packages

        result = check_all_packages()
        for key in ("sam2", "sam3"):
            val = result[key]
            assert isinstance(val, tuple)
            assert len(val) == 2
            assert isinstance(val[0], bool)
            assert val[1] is None or isinstance(val[1], str)
