"""Generic adapter for user datasets following MVTec-like contracts."""

from grdnet.data.adapters.mvtec import MvtecLikeAdapter


class CustomMvtecLikeAdapter(MvtecLikeAdapter):
    """Alias adapter kept explicit for extension points."""
