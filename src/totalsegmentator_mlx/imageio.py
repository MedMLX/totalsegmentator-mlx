"""Typed NIfTI writing with a parallel gzip stream, and affine access."""

import io
import os
import struct
import zlib
from collections import deque
from collections.abc import Buffer
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import InvalidInputError
from numpy.typing import NDArray

_DEFLATE_CHUNK = 1 << 20
_DEFLATE_WINDOW = 1 << 15
_DEFLATE_WORKERS = min(4, os.cpu_count() or 1)


class ImageWriter(Protocol):
    """Declare the NiBabel methods used to serialize a NIfTI image."""

    def to_filename(self, filename: Path) -> None:
        """Write the image to a filename using its format."""
        ...

    def to_stream(self, io_obj: io.IOBase) -> None:
        """Serialize the complete single-file NIfTI to a stream."""
        ...


def _deflate_chunk(chunk: bytes, window: bytes, *, final: bool) -> bytes:
    """Compress a byte-aligned part of the gzip member's raw deflate stream."""
    compressor = zlib.compressobj(1, zlib.DEFLATED, -zlib.MAX_WBITS, zdict=window)
    return compressor.compress(chunk) + compressor.flush(
        zlib.Z_FINISH if final else zlib.Z_SYNC_FLUSH
    )


class _ParallelGzip(io.RawIOBase):
    """Stream one gzip member with at most four 1 MiB chunks in flight."""

    def __init__(self, stream: io.BufferedWriter, pool: ThreadPoolExecutor) -> None:
        self._stream = stream
        self._pool = pool
        self._pending: deque[Future[bytes]] = deque()
        self._buffer = bytearray()
        self._window = b""
        self._checksum = 0
        self._size = 0
        # Match NiBabel's level 1 header with zero mtime and no filename.
        stream.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x04\xff")

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._size

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        """Let NiBabel handle forward padding when a seek needs to move."""
        if whence == io.SEEK_SET and offset == self._size:
            return offset
        raise io.UnsupportedOperation("gzip writes are sequential")

    def write(self, b: Buffer, /) -> int:
        """Fill fixed-size chunks that can be compressed independently."""
        data = memoryview(b).cast("B")
        size = len(data)
        while data:
            count = min(len(data), _DEFLATE_CHUNK - len(self._buffer))
            self._buffer.extend(data[:count])
            data = data[count:]
            if len(self._buffer) == _DEFLATE_CHUNK:
                self._submit(final=False)
        self._size += size
        return size

    def _submit(self, *, final: bool) -> None:
        chunk = bytes(self._buffer)
        self._buffer.clear()
        # The previous 32 KiB lets deflate matches cross chunk boundaries.
        self._pending.append(self._pool.submit(_deflate_chunk, chunk, self._window, final=final))
        self._window = chunk[-_DEFLATE_WINDOW:]
        self._checksum = zlib.crc32(chunk, self._checksum)
        if len(self._pending) == _DEFLATE_WORKERS:
            self._stream.write(self._pending.popleft().result())

    def finish(self) -> None:
        """Finish the deflate stream and write its required gzip trailer."""
        self._submit(final=True)
        while self._pending:
            self._stream.write(self._pending.popleft().result())
        self._stream.write(struct.pack("<II", self._checksum, self._size & 0xFFFFFFFF))


def save_nifti(image: nib.Nifti1Image, path: Path) -> None:
    """Write a NIfTI image, streaming array-backed ``.nii.gz`` files as one gzip member."""
    writer = cast(ImageWriter, image)
    # NiBabel reads a file-backed proxy before opening the destination for writing.
    if not path.name.lower().endswith(".nii.gz") or not isinstance(image.dataobj, np.ndarray):
        writer.to_filename(path)
        return
    with path.open("wb") as stream, ThreadPoolExecutor(max_workers=_DEFLATE_WORKERS) as pool:
        compressor = _ParallelGzip(stream, pool)
        writer.to_stream(compressor)
        compressor.finish()
    image.set_filename(str(path))


def spatial_affine(image: nib.Nifti1Image) -> NDArray[np.float64]:
    """Return the 4x4 voxel-to-world affine as float64, without converting units.

    The caller must interpret or validate the header's spatial units.
    """
    affine = image.affine
    if affine is None:
        raise InvalidInputError("NIfTI image requires a spatial affine")
    return np.asarray(affine, dtype=np.float64)
