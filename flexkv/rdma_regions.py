"""Optional CPU read regions for transfer engines that select one NIC per request."""


def split_read_regions(src_ptrs, dst_ptrs, sizes, region_bytes):
    if region_bytes == 0:
        return src_ptrs, dst_ptrs, sizes
    if region_bytes < 0 or not len(src_ptrs) == len(dst_ptrs) == len(sizes):
        raise ValueError("invalid CPU RDMA region size or vector lengths")
    sources, destinations, lengths = [], [], []
    for src, dst, size in zip(src_ptrs, dst_ptrs, sizes, strict=True):
        if size <= 0:
            raise ValueError("CPU RDMA read size must be positive")
        for offset in range(0, size, region_bytes):
            sources.append(src + offset)
            destinations.append(dst + offset)
            lengths.append(min(region_bytes, size - offset))
    return sources, destinations, lengths
