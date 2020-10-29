#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
"""Merge datasets
"""

import os
from pathlib import Path
import shutil

import configargparse
from tqdm import tqdm
from datumaro.components.project import Environment, Project
from datumaro.components.operations import IntersectMerge


# list of temporary directories we created during the merge
# which will have to be cleaned up after we're done
cleanup_paths = []


def detect_format(dataset_dir, args):
    """try to figure out what type of dataset we may have"""

    # datumaro gets confused with 2 other formats, but the others
    # don't have the `.datumaro/config.yaml` file.
    if dataset_dir.joinpath(".datumaro", "config.yaml").exists():
        return "datumaro"

    env = Environment()
    matches = []
    for format_name in env.importers.items:
        importer = env.make_importer(format_name)
        try:
            match = importer.detect(os.fspath(dataset_dir))
            if match:
                matches.append((format_name, importer))
        except NotImplementedError:
            pass

    if args.verbose:
        print(f"Found {matches} for {dataset_dir}")

    if len(matches) == 1:
        return matches[0]

    assert False, "Unrecognized dataset format"
    return None


def project_path(dataset_path, suffix):
    """create predictable temporary project path name for dataset"""
    return Path("merge_" + dataset_path.stem).with_suffix(suffix)


def convert_to_datumaro(dataset_path, args):
    """datum project import -f {dataset_format} -i {dataset_path} -o {project_dir}"""

    dataset_format = detect_format(dataset_path, args)
    if dataset_format == "datumaro":
        return dataset_path

    tmp_project_dir = project_path(dataset_path, ".datumaro")
    cleanup_paths.append(tmp_project_dir)

    print(f"Converting {dataset_path} to datumaro format")
    project = Project.import_from(dataset_path, dataset_format)
    project.config.project_name = dataset_path.name
    project.config.project_dir = str(tmp_project_dir)

    print("Checking dataset...")
    _ = project.make_dataset()  # check dataset
    project.save()

    return tmp_project_dir


def filter_empty_frames(dataset_path, project_dir):
    """datum project filter -p {project_dir} -m i+a '//*'"""

    print(f"Removing empty frames from {dataset_path}")
    project = Project.load(project_dir)
    dataset = project.make_dataset()

    tmp_project_dir = project_path(dataset_path, ".filtered")
    cleanup_paths.append(tmp_project_dir)

    dataset.filter_project(
        save_dir=tmp_project_dir,
        filter_expr="//*",
        filter_annotations=True,
        remove_empty=True,
    )

    # catch empty datasets early?
    if not tmp_project_dir.joinpath("dataset", "annotations", "default.json").exists():
        print(f"No annotated frames found in {dataset_path}, dropping dataset")
        return None

    return tmp_project_dir


def reindex(dataset_path, project_dir, start_index):
    """datum project transform -p {project_dir} -t reindex -- -s {start_index}"""

    print(f"Reindexing {dataset_path}")
    project = Project.load(project_dir)
    dataset = project.make_dataset()

    tmp_project_dir = project_path(dataset_path, ".reindexed")
    cleanup_paths.append(tmp_project_dir)

    reindex_xfrm = dataset.env.transforms.get("reindex")
    dataset.transform_project(
        method=reindex_xfrm, save_dir=tmp_project_dir, start=start_index
    )

    return tmp_project_dir, len(dataset)


def merge(cleaned_datasets, output):
    """datum merge -o {output} {project_dirs}"""

    print(f"Merging datasets to {output}/")
    projects = [Project.load(p) for p in cleaned_datasets]
    datasets = [p.make_dataset() for p in projects]

    merged_project_dir = Path(output)

    # perform the merge
    merge_config = IntersectMerge.Conf(
        pairwise_dist=0.25,
        groups=[],
        output_conf_thresh=0.0,
        quorum=0,
    )
    merged_dataset = IntersectMerge(conf=merge_config)(datasets)

    merged_project = Project()
    output_dataset = merged_project.make_dataset()
    output_dataset.define_categories(merged_dataset.categories())
    merged_dataset = output_dataset.update(merged_dataset)
    merged_dataset.save(save_dir=merged_project_dir)


def main():
    """Merge datasets with Datumaro"""

    parser = configargparse.ArgParser(default_config_files=["~/.opentpod-tools"])
    # *** these have to be defined if we want a common config file, not ideal...
    parser.add_argument("--url", help=configargparse.SUPPRESS)
    parser.add_argument("--username", help=configargparse.SUPPRESS)
    parser.add_argument("--password", help=configargparse.SUPPRESS)
    # ***
    parser.add_argument("-c", "--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Extra verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Leave temporary datasets around for debugging",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="merged",
        help="Merged dataset name (defaults to 'merged')",
    )
    parser.add_argument(
        "dataset", type=Path, nargs="+", help="paths of datasets to merge"
    )
    args = parser.parse_args()

    try:
        # Run a couple of cleanup operations before trying to merge
        cleaned_datasets = []
        start_index = 0
        for dataset in args.dataset:
            # convert all datasets to datumaro format
            project_dir = convert_to_datumaro(dataset, args)

            # remove frames with no annotations
            filtered = filter_empty_frames(dataset, project_dir)
            if filtered is None:
                continue

            # reindex to avoid colliding item ids
            reindexed, items = reindex(dataset, filtered, start_index=start_index)
            cleaned_datasets.append(reindexed)
            start_index += items

        # And then do the actual merge operation
        merge(cleaned_datasets, output=args.output)

    finally:
        if not args.debug:
            print("Cleaning up temporary directories")
            with tqdm(cleanup_paths, leave=False) as pbar:
                for path in pbar:
                    pbar.set_description(f"deleting {path}")
                    shutil.rmtree(path)


if __name__ == "__main__":
    main()
