"""High-performance, release-driven, dependency-free PLY codec for Polyxios."""

from pathlib import Path
from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".ply"

MAX_CONNECTIVITY_INDEX: int = 2**31 - 1

_PLY_DTYPE: dict[str, str] = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float32": "f4",
    "float64": "f8",
}


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a PLY file and return a PolyData instance."""
    path = Path(path)
    file_size = path.stat().st_size

    with open(path, "rb") as fh:
        header, header_end_offset = _parse_header(fh)

    fmt = header["format"]
    elements = header["elements"]

    vert_elem = next((e for e in elements if e["name"] == "vertex"), None)
    face_elem = next((e for e in elements if e["name"] == "face"), None)

    n_verts = vert_elem["count"] if vert_elem else 0
    n_faces = face_elem["count"] if face_elem else 0

    conn_estimate = n_faces * 4
    validate_header(n_verts, n_faces, conn_estimate, file_size)

    if fmt == "ascii":
        if lazy:
            raise LazyReadError("PLY ASCII format does not support lazy reads.")
        return _read_ascii(path, header, header_end_offset)

    little_endian = fmt == "binary_little_endian"

    # NOTE: Genuine zero-copy lazy loading is structurally impossible for PLY formats
    # due to the variable-length nature of face index lists. mmap usage here only causes
    # memory leak deadlocks on tracebacks. We fall back to a full strict read.
    return _read_binary(path, header, header_end_offset, little_endian)


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to a PLY file."""
    path = Path(path)
    binary: bool = bool(opts.get("binary", True))
    endian: str = str(opts.get("endian", "little"))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    lines: list[bytes] = [b"ply"]

    if binary:
        fmt_str = f"binary_{endian}_endian"
        lines.append(f"format {fmt_str} 1.0".encode())
    else:
        lines.append(b"format ascii 1.0")

    lines.append(b"comment Written by polyxios")

    lines.append(f"element vertex {n_verts}".encode())
    lines.append(b"property double x")
    lines.append(b"property double y")
    lines.append(b"property double z")

    for name, arr in poly.vertex_attrs.items():
        dt_str = _np_to_ply_type(arr.dtype)
        if arr.ndim == 2:
            lines.extend(
                f"property {dt_str} {name}_{ci}".encode() for ci in range(arr.shape[1])
            )
        else:
            lines.append(f"property {dt_str} {name}".encode())

    lines.append(f"element face {n_elems}".encode())
    lines.append(b"property list uchar int vertex_indices")

    for name, arr in poly.element_attrs.items():
        dt_str = _np_to_ply_type(arr.dtype)
        lines.append(f"property {dt_str} {name}".encode())

    lines.append(b"end_header")

    header_bytes = b"\n".join(lines) + b"\n"

    with open(path, "wb") as fh:
        fh.write(header_bytes)

        if binary:
            endian_chr = "<" if endian == "little" else ">"
            for vi in range(n_verts):
                fh.write(
                    np.asarray(
                        poly.vertices[vi], dtype=np.dtype(endian_chr + "f8")
                    ).tobytes()
                )
                for arr in poly.vertex_attrs.values():
                    row = arr[vi]
                    val = np.asarray(row).ravel()
                    dt = np.dtype(endian_chr + _np_short_dtype(val.dtype))
                    fh.write(val.astype(dt).tobytes())

            count_dt = np.dtype("u1")
            idx_dt = np.dtype(endian_chr + "i4")
            for i in range(n_elems):
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                fh.write(np.array([e - s], dtype=count_dt).tobytes())
                fh.write(poly.connectivity[s:e].astype(idx_dt).tobytes())
                for arr in poly.element_attrs.values():
                    val = np.asarray(arr[i]).ravel()
                    dt = np.dtype(endian_chr + _np_short_dtype(val.dtype))
                    fh.write(val.astype(dt).tobytes())
        else:
            for vi in range(n_verts):
                row = [
                    f"{poly.vertices[vi, 0]:.10g}",
                    f"{poly.vertices[vi, 1]:.10g}",
                    f"{poly.vertices[vi, 2]:.10g}",
                ]
                for arr in poly.vertex_attrs.values():
                    if arr.ndim == 2:
                        row.extend(f"{arr[vi, ci]:.10g}" for ci in range(arr.shape[1]))
                    else:
                        row.append(f"{arr[vi]:.10g}")
                fh.write((" ".join(row) + "\n").encode())

            for i in range(n_elems):
                s, e = int(poly.offsets[i]), int(poly.offsets[i + 1])
                parts = [str(e - s)] + [str(int(v)) for v in poly.connectivity[s:e]]
                parts.extend(f"{arr[i]:.10g}" for arr in poly.element_attrs.values())
                fh.write((" ".join(parts) + "\n").encode())


def _read_ascii(path: Path, header: dict, header_end_offset: int) -> PolyData:
    vert_elem = next((e for e in header["elements"] if e["name"] == "vertex"), None)
    face_elem = next((e for e in header["elements"] if e["name"] == "face"), None)

    n_verts = vert_elem["count"] if vert_elem else 0
    n_faces = face_elem["count"] if face_elem else 0

    with open(path, "rb") as fh:
        fh.seek(header_end_offset)
        lines = fh.read().decode("ascii", errors="replace").splitlines()

    idx = 0
    vertices = np.zeros((n_verts, 3), dtype=np.float64)
    extra_vert_props: dict[str, list] = {}

    if vert_elem:
        props = vert_elem["properties"]
        coord_map = {"x": 0, "y": 1, "z": 2}
        for vi in range(n_verts):
            vals = lines[idx].split()
            idx += 1
            for pi, (pname, _) in enumerate(props):
                if pname in coord_map:
                    vertices[vi, coord_map[pname]] = float(vals[pi])
                else:
                    extra_vert_props.setdefault(pname, []).append(float(vals[pi]))

    vertex_attrs = {k: np.array(v) for k, v in extra_vert_props.items()}

    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    extra_face_props: dict[str, list] = {}

    if face_elem:
        face_props = face_elem["properties"]
        for _fi in range(n_faces):
            vals = lines[idx].split()
            idx += 1
            vi = 0
            for pname, ptype in face_props:
                if isinstance(ptype, tuple) and ptype[0] == "list":
                    cnt = int(vals[vi])
                    vi += 1
                    for _ in range(cnt):
                        conn_list.append(int(vals[vi]))
                        vi += 1
                    offsets_list.append(offsets_list[-1] + cnt)
                else:
                    extra_face_props.setdefault(pname, []).append(float(vals[vi]))
                    vi += 1

    poly_code = ELEMENT_TYPES["polygon"]
    tri_code = ELEMENT_TYPES["triangle"]
    quad_code = ELEMENT_TYPES["quad"]

    types_list: list[int] = []
    n_elems = len(offsets_list) - 1
    for i in range(n_elems):
        n = offsets_list[i + 1] - offsets_list[i]
        if n == 3:
            types_list.append(tri_code)
        elif n == 4:
            types_list.append(quad_code)
        else:
            types_list.append(poly_code)

    element_attrs = {k: np.array(v) for k, v in extra_face_props.items()}

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def _read_binary(
    path: Path, header: dict, header_end_offset: int, little_endian: bool
) -> PolyData:
    with open(path, "rb") as fh:
        fh.seek(header_end_offset)
        data = fh.read()

    mv = memoryview(data)
    return _decode_binary(mv, header, little_endian)


def _decode_binary(
    mv: memoryview,
    header: dict,
    little_endian: bool,
) -> PolyData:
    endian = "<" if little_endian else ">"
    pos = 0

    vertices = np.zeros((0, 3), dtype=np.float64)
    vertex_attrs: dict[str, np.ndarray] = {}
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    element_attrs: dict[str, list] = {}

    for elem in header["elements"]:
        ename = elem["name"]
        count = elem["count"]
        props = elem["properties"]

        if ename == "vertex":
            dtype_fields: list[tuple[str, str]] = []
            for pname, ptype in props:
                dtype_fields.append((pname, endian + _PLY_DTYPE[ptype]))
            dt = np.dtype(dtype_fields)
            nbytes = count * dt.itemsize

            raw = mv[pos : pos + nbytes]
            rec = np.frombuffer(raw, dtype=dt)
            pos += nbytes

            coords = np.zeros((count, 3), dtype=np.float64)
            coord_map = {"x": 0, "y": 1, "z": 2}
            for pname, _ in props:
                if pname in coord_map:
                    coords[:, coord_map[pname]] = rec[pname].astype(np.float64)
                else:
                    vertex_attrs[pname] = np.array(rec[pname])
            vertices = coords

        elif ename == "face":
            # Map decoders to sequentially parse face properties as they appeared in header
            prop_decoders = []
            for pname, ptype in props:
                if isinstance(ptype, tuple) and ptype[0] == "list":
                    count_dt = np.dtype(endian + _PLY_DTYPE.get(ptype[1], "u1"))
                    idx_dt = np.dtype(endian + _PLY_DTYPE.get(ptype[2], "i4"))
                    prop_decoders.append(("list", pname, count_dt, idx_dt))
                else:
                    prop_decoders.append(
                        ("scalar", pname, np.dtype(endian + _PLY_DTYPE[ptype]))
                    )

            for p_mode, pname, *_ in prop_decoders:
                if p_mode == "scalar":
                    element_attrs[pname] = []

            for _ in range(count):
                for p_info in prop_decoders:
                    if p_info[0] == "list":
                        _, pname, count_dt, idx_dt = p_info
                        cnt_size = count_dt.itemsize
                        cnt = int(
                            np.frombuffer(mv[pos : pos + cnt_size], dtype=count_dt)[0]
                        )
                        pos += cnt_size

                        idx_size = idx_dt.itemsize
                        nbytes = cnt * idx_size

                        if pos + nbytes > len(mv):
                            raise CodecError(
                                "Unexpected EOF while parsing face indices."
                            )

                        indices = np.frombuffer(
                            mv[pos : pos + nbytes], dtype=idx_dt
                        ).astype(np.int32)
                        pos += nbytes

                        # Only aggregate if it explicitly defines vertex connectivity
                        if pname in ("vertex_indices", "vertex_index"):
                            conn_list.extend(indices.tolist())
                            offsets_list.append(offsets_list[-1] + cnt)

                    elif p_info[0] == "scalar":
                        _, pname, dt = p_info
                        esize = dt.itemsize
                        val = np.frombuffer(mv[pos : pos + esize], dtype=dt)[0]
                        pos += esize
                        element_attrs[pname].append(float(val))

        else:
            pos = _skip_binary_element(mv, pos, count, props, endian)

    n_elems = len(offsets_list) - 1
    tri_code = ELEMENT_TYPES["triangle"]
    quad_code = ELEMENT_TYPES["quad"]
    poly_code = ELEMENT_TYPES["polygon"]
    types_list: list[int] = []

    for i in range(n_elems):
        n = offsets_list[i + 1] - offsets_list[i]
        if n == 3:
            types_list.append(tri_code)
        elif n == 4:
            types_list.append(quad_code)
        else:
            types_list.append(poly_code)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs={k: np.array(v) for k, v in element_attrs.items()},
    )


def _skip_binary_element(
    mv: memoryview,
    pos: int,
    count: int,
    props: list[tuple[str, Any]],
    endian: str,
) -> int:
    for _ in range(count):
        for _, ptype in props:
            if isinstance(ptype, tuple) and ptype[0] == "list":
                count_dt = np.dtype(endian + _PLY_DTYPE.get(ptype[1], "u1"))
                cnt_size = count_dt.itemsize
                cnt = int(np.frombuffer(mv[pos : pos + cnt_size], dtype=count_dt)[0])
                pos += cnt_size

                idx_size = np.dtype(endian + _PLY_DTYPE.get(ptype[2], "i4")).itemsize
                pos += cnt * idx_size
            else:
                pos += np.dtype(endian + _PLY_DTYPE.get(ptype, "f4")).itemsize
    return pos


def _parse_header(fh: object) -> tuple[dict, int]:
    offset = 0

    first = fh.readline()  # type: ignore[union-attr]
    offset += len(first)
    if first.strip() != b"ply":
        raise CodecError("Not a PLY file (missing 'ply' header)")

    header: dict[str, Any] = {"format": "ascii", "elements": []}
    current_elem: dict[str, Any] | None = None

    while True:
        raw = fh.readline()  # type: ignore[union-attr]
        offset += len(raw)
        line = raw.decode("ascii", errors="replace").strip()

        if line == "end_header":
            break

        parts = line.split()
        if not parts:
            continue
        kw = parts[0]

        if kw == "format":
            header["format"] = parts[1]
        elif kw == "element":
            current_elem = {"name": parts[1], "count": int(parts[2]), "properties": []}
            header["elements"].append(current_elem)
        elif kw == "property" and current_elem is not None:
            if parts[1] == "list":
                current_elem["properties"].append(
                    (parts[4], ("list", parts[2], parts[3]))
                )
            else:
                current_elem["properties"].append((parts[2], parts[1]))

    return header, offset


def _np_to_ply_type(dtype: np.dtype) -> str:
    kind = dtype.kind
    size = dtype.itemsize
    if kind == "f":
        return "double" if size == 8 else "float"
    if kind in ("i", "u"):
        signed = kind == "i"
        mapping = {1: "char", 2: "short", 4: "int"}
        base = mapping.get(size, "int")
        return base if signed else "u" + base
    return "float"


def _np_short_dtype(dtype: np.dtype) -> str:
    kind = dtype.kind
    size = dtype.itemsize
    if kind == "f":
        return "f8" if size == 8 else "f4"
    if kind == "i":
        return f"i{size}"
    if kind == "u":
        return f"u{size}"
    return "f4"
