#! /usr/bin/env python

import os
import re
import sys
from argparse import ArgumentParser
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import onnx
from onnx.external_data_helper import uses_external_data
import onnx_graphsurgeon as gs

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

ONNX_STANDARD_DOMAINS = [
    'ai.onnx',
    'ai.onnx.ml',
    '',
]

SIZE_UNIT_FACTORS = {
    'kb': 1024,
    'mb': 1024 * 1024,
    'gb': 1024 * 1024 * 1024,
}


def _print_info(message: str, *, non_verbose: bool) -> None:
    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} {message}')


def _print_warn(message: str, *, non_verbose: bool) -> None:
    if not non_verbose:
        print(f'{Color.YELLOW}WARNING:{Color.RESET} {message}')


def _print_error(message: str) -> None:
    print(f'{Color.RED}ERROR:{Color.RESET} {message}')


def _parse_size_to_bytes(size: str) -> int:
    if size is None:
        raise ValueError('auto_split_max_size is required.')
    if isinstance(size, (int, float)):
        return int(float(size) * SIZE_UNIT_FACTORS['mb'])
    size_str = str(size).strip()
    match = re.fullmatch(r'(?i)\s*([0-9]*\.?[0-9]+)\s*([kmg]b)?\s*', size_str)
    if not match:
        raise ValueError(
            'auto_split_max_size must be like "100MB", "512KB", or "1GB".'
        )
    value = float(match.group(1))
    unit = (match.group(2) or 'mb').lower()
    factor = SIZE_UNIT_FACTORS.get(unit)
    if factor is None:
        raise ValueError(f'Unsupported unit for auto_split_max_size: {unit}')
    return int(value * factor)


def _tensorproto_nbytes(tensor: onnx.TensorProto) -> int:
    if tensor is None:
        return 0
    try:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    except Exception:
        np_dtype = None
    if np_dtype is None:
        return 0
    elem_size = np.dtype(np_dtype).itemsize
    num_elems = int(np.prod(tensor.dims)) if len(tensor.dims) > 0 else 0
    if num_elems == 0:
        try:
            field_name = onnx.helper.tensor_dtype_to_field(tensor.data_type)
            if hasattr(tensor, field_name):
                num_elems = len(getattr(tensor, field_name))
        except Exception:
            num_elems = 0
    return num_elems * elem_size


def _collect_initializer_sizes(onnx_graph: onnx.ModelProto) -> Dict[str, int]:
    initializer_sizes: Dict[str, int] = {}
    if onnx_graph is None:
        return initializer_sizes
    for initializer in onnx_graph.graph.initializer:
        if not initializer.name:
            continue
        try:
            initializer_sizes[initializer.name] = _tensorproto_nbytes(initializer)
        except Exception:
            initializer_sizes[initializer.name] = 0
    return initializer_sizes


def _collect_node_weight_keys(
    *,
    graph: gs.Graph,
    initializer_sizes: Dict[str, int],
) -> Tuple[List[List[str]], Dict[str, int]]:
    weight_sizes = dict(initializer_sizes)
    node_weight_keys: List[List[str]] = []
    for node in graph.nodes:
        keys: List[str] = []
        for inp in node.inputs:
            if isinstance(inp, gs.Constant):
                if isinstance(getattr(inp, 'values', None), np.ndarray):
                    key = f'const:{id(inp)}'
                    if key not in weight_sizes:
                        weight_sizes[key] = int(inp.values.nbytes)
                    keys.append(key)
                continue
            name = getattr(inp, 'name', '')
            if name and name in initializer_sizes:
                keys.append(name)
        node_weight_keys.append(keys)
    return node_weight_keys, weight_sizes


def _auto_partition_ranges(
    *,
    node_weight_keys: List[List[str]],
    weight_sizes: Dict[str, int],
    max_size_bytes: int,
    reachable_node_indices: Optional[set] = None,
) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    if max_size_bytes <= 0 or not node_weight_keys:
        return ranges
    current_keys: set = set()
    current_bytes = 0
    start_idx = 0
    for idx, keys in enumerate(node_weight_keys):
        new_bytes = 0
        for key in keys:
            if key not in current_keys:
                new_bytes += weight_sizes.get(key, 0)
                current_keys.add(key)
        current_bytes += new_bytes
        if current_bytes >= max_size_bytes and idx > start_idx:
            if reachable_node_indices is not None and idx not in reachable_node_indices:
                continue
            ranges.append((start_idx, idx))
            start_idx = idx + 1
            current_keys = set()
            current_bytes = 0
    if start_idx <= len(node_weight_keys) - 1:
        ranges.append((start_idx, len(node_weight_keys) - 1))
    return ranges


def _collect_reachable_node_indices(
    graph: gs.Graph,
    initializer_names: Optional[set] = None,
) -> set:
    reachable_nodes: set = set()
    reachable_vars: set = set()
    initializer_names = initializer_names or set()
    for graph_input in graph.inputs:
        name = getattr(graph_input, 'name', '')
        if name and name not in initializer_names:
            reachable_vars.add(name)
    for idx, node in enumerate(graph.nodes):
        is_reachable = False
        for inp in node.inputs:
            if isinstance(inp, gs.Variable):
                name = getattr(inp, 'name', '')
                if name in reachable_vars and name not in initializer_names:
                    is_reachable = True
                    break
        if is_reachable:
            reachable_nodes.add(idx)
            for out in node.outputs:
                name = getattr(out, 'name', '')
                if name:
                    reachable_vars.add(name)
    return reachable_nodes


def _collect_constant_only_node_indices(
    graph: gs.Graph,
    initializer_names: Optional[set] = None,
) -> set:
    initializer_names = initializer_names or set()
    const_only_nodes: set = set()
    for idx, node in enumerate(graph.nodes):
        has_variable_input = False
        for inp in node.inputs:
            if isinstance(inp, gs.Constant):
                continue
            name = getattr(inp, 'name', '')
            if name and name not in initializer_names:
                has_variable_input = True
                break
        if not has_variable_input:
            const_only_nodes.add(idx)
    return const_only_nodes


def _estimate_partition_weight_bytes(
    *,
    ranges: List[Tuple[int, int]],
    node_weight_keys: List[List[str]],
    weight_sizes: Dict[str, int],
) -> List[int]:
    partition_sizes: List[int] = []
    for start_idx, end_idx in ranges:
        seen: set = set()
        total_bytes = 0
        for idx in range(start_idx, end_idx + 1):
            for key in node_weight_keys[idx]:
                if key not in seen:
                    total_bytes += weight_sizes.get(key, 0)
                    seen.add(key)
        partition_sizes.append(total_bytes)
    return partition_sizes


def _build_partition_io(
    *,
    graph: gs.Graph,
    ranges: List[Tuple[int, int]],
    const_only_nodes: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if not ranges:
        return []
    const_only_nodes = const_only_nodes or set()
    producer_by_tensor: Dict[str, int] = {}
    consumers_by_tensor: Dict[str, set] = {}
    graph_output_names = [o.name for o in graph.outputs if o.name]
    for idx, node in enumerate(graph.nodes):
        for out in node.outputs:
            name = getattr(out, 'name', '')
            if name:
                producer_by_tensor[name] = idx
        for inp in node.inputs:
            if isinstance(inp, gs.Constant):
                continue
            name = getattr(inp, 'name', '')
            if not name:
                continue
            consumers_by_tensor.setdefault(name, set()).add(idx)

    partitions: List[Dict[str, Any]] = []
    for start_idx, end_idx in ranges:
        node_idx_set = set(range(start_idx, end_idx + 1))
        part_inputs: set = set()
        part_outputs: set = set()
        for idx in node_idx_set:
            node = graph.nodes[idx]
            for inp in node.inputs:
                if isinstance(inp, gs.Constant):
                    continue
                name = getattr(inp, 'name', '')
                if not name:
                    continue
                producer_idx = producer_by_tensor.get(name)
                if producer_idx is None or producer_idx not in node_idx_set:
                    if producer_idx is not None and producer_idx in const_only_nodes:
                        continue
                    part_inputs.add(name)
            for out in node.outputs:
                name = getattr(out, 'name', '')
                if not name:
                    continue
                consumers = consumers_by_tensor.get(name, set())
                if name in graph_output_names or any(c not in node_idx_set for c in consumers):
                    if idx in const_only_nodes and name not in graph_output_names:
                        continue
                    part_outputs.add(name)
        partitions.append({
            'inputs': sorted(part_inputs),
            'outputs': sorted(part_outputs),
            'node_count': end_idx - start_idx + 1,
            'start_idx': start_idx,
            'end_idx': end_idx,
        })
    return partitions


def _merge_ranges_with_missing_io(
    *,
    graph: gs.Graph,
    ranges: List[Tuple[int, int]],
    const_only_nodes: Optional[set] = None,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    if not ranges:
        return ranges, []
    ranges = list(ranges)
    const_only_nodes = const_only_nodes or set()
    while True:
        partitions = _build_partition_io(
            graph=graph,
            ranges=ranges,
            const_only_nodes=const_only_nodes,
        ) or []
        if all(part['inputs'] and part['outputs'] for part in partitions):
            return ranges, partitions
        if len(ranges) <= 1:
            return ranges, partitions
        merged = False
        for idx, part in enumerate(partitions):
            if not part['inputs'] or not part['outputs']:
                if idx > 0:
                    ranges[idx - 1] = (ranges[idx - 1][0], ranges[idx][1])
                    del ranges[idx]
                else:
                    ranges[idx] = (ranges[idx][0], ranges[idx + 1][1])
                    del ranges[idx + 1]
                merged = True
                break
        if not merged:
            return ranges, partitions


def model_uses_external_data(input_onnx_file_path: str) -> bool:
    model = onnx.load(input_onnx_file_path, load_external_data=False)
    def iter_tensors_in_graph(g):
        for t in g.initializer:
            yield t
        for t in g.sparse_initializer:
            yield t
        for n in g.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    yield a.t
                elif a.type == onnx.AttributeProto.TENSORS:
                    for t in a.tensors:
                        yield t
                elif a.type == onnx.AttributeProto.GRAPH:
                    yield from iter_tensors_in_graph(a.g)
                elif a.type == onnx.AttributeProto.GRAPHS:
                    for sg in a.graphs:
                        yield from iter_tensors_in_graph(sg)
    return any(uses_external_data(t) for t in iter_tensors_in_graph(model.graph))

def extraction(
    input_op_names: List[str],
    output_op_names: List[str],
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    has_external_data: Optional[bool] = False,
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:

    """
    Parameters
    ----------
    input_op_names: List[str]
        List of OP names to specify for the input layer of the model.\n\
        e.g. ['aaa','bbb','ccc']

    output_op_names: List[str]
        List of OP names to specify for the output layer of the model.\n\
        e.g. ['ddd','eee','fff']

    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If not specified, .onnx is not output.\n\
        Default: ''

    has_external_data: Optional[bool]
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    extracted_graph: onnx.ModelProto
        Extracted onnx ModelProto
    """

    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    if not input_op_names:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One or more input_op_names must be specified.'
        )
        sys.exit(1)

    if not output_op_names:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One or more output_op_names must be specified.'
        )
        sys.exit(1)

    # Load
    graph = None
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
        has_external_data = model_uses_external_data(input_onnx_file_path)

    # Acquisition of Node with custom domain
    custom_domain_check_onnx_nodes = []
    custom_domain_check_onnx_nodes = \
        custom_domain_check_onnx_nodes + \
            [
                node for node in onnx_graph.graph.node \
                    if node.domain not in ONNX_STANDARD_DOMAINS
            ]

    # domain, ir_version
    domain: str = onnx_graph.domain
    ir_version: int = onnx_graph.ir_version
    meta_data = {'domain': domain, 'ir_version': ir_version}
    metadata_props = None
    if hasattr(onnx_graph, 'metadata_props'):
        metadata_props = onnx_graph.metadata_props

    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()

    # Check if Graph contains a custom domain (custom module)
    contains_custom_domain = len(
        [
            domain \
                for domain in graph.import_domains \
                    if domain.domain not in ONNX_STANDARD_DOMAINS
        ]
    ) > 0

    # Extraction of input OP and output OP
    graph_node_inputs = [
        graph_node \
            for graph_node in graph.nodes \
                for graph_nodes_input in graph_node.inputs \
                    if graph_nodes_input.name in input_op_names
    ]
    graph_node_outputs = [
        graph_node \
            for graph_node in graph.nodes \
                for graph_nodes_output in graph_node.outputs \
                    if graph_nodes_output.name in output_op_names
    ]

    # Init graph INPUT/OUTPUT
    graph.inputs.clear()
    graph.outputs.clear()

    # Update graph INPUT/OUTPUT
    input_tmp = []
    for graph_node in graph_node_inputs:
        for graph_node_input in graph_node.inputs:
            if graph_node_input.shape \
                and graph_node_input not in [i for i in input_tmp] \
                and hasattr(graph_node_input, 'name') \
                and graph_node_input.name in [i for i in input_op_names]:
                input_tmp.append(graph_node_input)
    graph.inputs = input_tmp

    graph.outputs = [
        graph_node_output \
            for graph_node in graph_node_outputs \
                for graph_node_output in graph_node.outputs
    ]

    # Cleanup
    graph.cleanup(remove_unused_graph_inputs=True).toposort()

    # Shape Estimation
    extracted_graph = None
    try:
        exported_onnx_graph = gs.export_onnx(graph, do_type_check=False, **meta_data)
        if metadata_props is not None:
            exported_onnx_graph.metadata_props.extend(metadata_props)
        extracted_graph = exported_onnx_graph
    except Exception as e:
        extracted_graph = gs.export_onnx(graph, do_type_check=False, **meta_data)
        if metadata_props is not None:
            extracted_graph.metadata_props.extend(metadata_props)
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )

    ## 5. Restore a node's custom domain
    if contains_custom_domain:
        extracted_graph_nodes = extracted_graph.graph.node
        for extracted_graph_node in extracted_graph_nodes:
            for custom_domain_check_onnx_node in custom_domain_check_onnx_nodes:
                if extracted_graph_node.name == custom_domain_check_onnx_node.name:
                    extracted_graph_node.domain = custom_domain_check_onnx_node.domain

    # Save
    if output_onnx_file_path:
        onnx.save(extracted_graph, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    return extracted_graph


def split(
    *,
    input_onnx_file_path: str,
    auto_split_max_size: str = '100MB',
    output_dir: Optional[str] = None,
    non_verbose: bool = False,
) -> List[str]:
    """
    Automatically splits an ONNX model by estimated weight size and saves each partition.

    The flow mirrors the auto-split algorithm from onnx2tf:
    1) Estimate initializer (weight) sizes.
    2) Collect per-node weight keys.
    3) Compute partition ranges using reachable/constant-only nodes.
    4) Merge ranges with missing inputs/outputs.
    5) Extract and write ONNX for each partition.

    Parameters
    ----------
    input_onnx_file_path: str
        Path to the input ONNX file. Required.

    auto_split_max_size: str
        Target partition size. Supports KB/MB/GB.
        Examples: "512KB", "100MB", "1GB".
        If the unit is omitted, MB is assumed.
        Default: "100MB".

    output_dir: Optional[str]
        Output directory. If not specified, outputs are written next to the input ONNX.

    non_verbose: bool
        If True, suppress informational logs.

    Returns
    -------
    List[str]
        List of generated ONNX partition paths.
        Output filenames follow {input_name}_{zero-padded 4-digit index}.onnx.

    Notes
    -----
    - Partition boundaries are based on estimated weight size and may not match file size.
    - If a partition is missing inputs or outputs, the model is saved as a single part.
    """
    if not input_onnx_file_path:
        _print_error('input_onnx_file_path is required.')
        sys.exit(1)
    if not os.path.isfile(input_onnx_file_path):
        _print_error(f'Input ONNX file not found: {input_onnx_file_path}')
        sys.exit(1)

    try:
        max_size_bytes = _parse_size_to_bytes(auto_split_max_size)
    except ValueError as ex:
        _print_error(str(ex))
        sys.exit(1)
    if max_size_bytes <= 0:
        _print_error('auto_split_max_size must be greater than 0.')
        sys.exit(1)

    if output_dir is None or output_dir == '':
        output_dir = os.path.dirname(os.path.abspath(input_onnx_file_path))
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_onnx_file_path))[0]
    onnx_graph = onnx.load(input_onnx_file_path)
    has_external_data = model_uses_external_data(input_onnx_file_path)

    graph = gs.import_onnx(onnx_graph)
    try:
        graph.toposort()
    except Exception:
        pass

    if not graph.nodes:
        output_path = os.path.join(output_dir, f'{base_name}_0001.onnx')
        onnx.save(onnx_graph, output_path)
        _print_info('Model has no nodes. Saved as a single part.', non_verbose=non_verbose)
        return [output_path]

    initializer_sizes = _collect_initializer_sizes(onnx_graph)
    node_weight_keys, weight_sizes = _collect_node_weight_keys(
        graph=graph,
        initializer_sizes=initializer_sizes,
    )
    const_only_nodes = _collect_constant_only_node_indices(
        graph,
        initializer_names=set(initializer_sizes.keys()),
    )
    reachable_node_indices = _collect_reachable_node_indices(
        graph,
        initializer_names=set(initializer_sizes.keys()),
    )
    ranges = _auto_partition_ranges(
        node_weight_keys=node_weight_keys,
        weight_sizes=weight_sizes,
        max_size_bytes=max_size_bytes,
        reachable_node_indices=reachable_node_indices,
    )
    if len(ranges) > 1:
        ranges, partitions = _merge_ranges_with_missing_io(
            graph=graph,
            ranges=ranges,
            const_only_nodes=const_only_nodes,
        )
    else:
        partitions = _build_partition_io(
            graph=graph,
            ranges=ranges,
            const_only_nodes=const_only_nodes,
        )

    if not partitions:
        _print_warn('Auto split failed to determine partitions. Saving as single part.', non_verbose=non_verbose)
        output_path = os.path.join(output_dir, f'{base_name}_0001.onnx')
        onnx.save(onnx_graph, output_path)
        return [output_path]
    if any((not part['inputs'] or not part['outputs']) for part in partitions):
        _print_warn(
            'Auto split produced partitions with missing inputs or outputs. Saving as single part.',
            non_verbose=non_verbose,
        )
        output_path = os.path.join(output_dir, f'{base_name}_0001.onnx')
        onnx.save(onnx_graph, output_path)
        return [output_path]

    partition_sizes = _estimate_partition_weight_bytes(
        ranges=ranges,
        node_weight_keys=node_weight_keys,
        weight_sizes=weight_sizes,
    )

    _print_info('Auto model partitioning enabled', non_verbose=non_verbose)
    _print_info(
        f'Target partition size (estimated weights): {max_size_bytes / (1024 * 1024):.2f} MB',
        non_verbose=non_verbose,
    )
    for idx, part in enumerate(partitions):
        size_mb = partition_sizes[idx] / (1024 * 1024)
        _print_info(
            f'  part {idx+1}: nodes={part["node_count"]}, '
            f'est_weights={size_mb:.2f} MB, '
            f'inputs={len(part["inputs"])}, outputs={len(part["outputs"])}',
            non_verbose=non_verbose,
        )

    output_paths: List[str] = []
    for idx, part in enumerate(partitions):
        output_path = os.path.join(output_dir, f'{base_name}_{idx+1:04d}.onnx')
        extraction(
            input_op_names=part['inputs'],
            output_op_names=part['outputs'],
            input_onnx_file_path=None,
            onnx_graph=onnx_graph,
            output_onnx_file_path=output_path,
            has_external_data=has_external_data,
            non_verbose=non_verbose,
        )
        output_paths.append(output_path)

    return output_paths


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_onnx_file',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default=None,
        help='Output directory. Default is the same directory as the input ONNX.'
    )
    parser.add_argument(
        '-s',
        '--auto_split_max_size',
        type=str,
        default='100MB',
        help='Target partition size. Supports KB/MB/GB. Default: 100MB.'
    )
    parser.add_argument(
        '-n',
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    split(
        input_onnx_file_path=args.input_onnx_file,
        auto_split_max_size=args.auto_split_max_size,
        output_dir=args.output_dir,
        non_verbose=args.non_verbose,
    )


if __name__ == '__main__':
    main()
