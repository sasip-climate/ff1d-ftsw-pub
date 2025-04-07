import importlib
import tomllib

import pytest

import ff1d_ftsw_pub.lib.data as libdata

data_root = "ff1d_ftsw_pub.data"
nb_figures = 1


def get_manifest_handle(subdir: importlib.resources.abc.Traversable):
    manifest_name = "manifest.toml"
    return subdir.joinpath(manifest_name)


def load_manifest(subdir: importlib.resources.abc.Traversable):
    with open(get_manifest_handle(subdir), "rb") as file:
        return tomllib.load(file)


def test_total_datasets():
    """Test there are as many data folders as expected."""
    data_path = importlib.resources.files(data_root)
    sub_folders = list(data_path.iterdir())
    assert len(sub_folders) == nb_figures


@pytest.mark.parametrize("subdir", importlib.resources.files(data_root).iterdir())
def test_all_manifests_exist(subdir: importlib.resources.abc.Traversable):
    """Test that all data folders have a manifest file."""
    assert get_manifest_handle(subdir).is_file()


@pytest.mark.parametrize("subdir", importlib.resources.files(data_root).iterdir())
def test_all_files_exist(subdir: importlib.resources.abc.Traversable):
    """Test all files listed in a manifest exist."""
    manifest = load_manifest(subdir)
    for filetype in manifest.values():
        for handle in filetype["files"]:
            assert subdir.joinpath(handle).is_file()


@pytest.mark.parametrize("subdir", importlib.resources.files(data_root).iterdir())
def test_files_are_loaded(subdir: importlib.resources.abc.Traversable):
    """Test that as many files listed in a manifest are loaded."""
    manifest = load_manifest(subdir)
    data = libdata.read_data(subdir)
    n_files = 0
    for file_format in manifest:
        for file in manifest[file_format]["files"]:
            n_files += 1
    assert n_files == len(data)
