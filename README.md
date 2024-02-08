# GEF plugins

## Setup

Prerequisite: [gef with CHERI support](https://github.com/CTSRD-CHERI/gef)

```sh
# assuming this directory is named gef-plugins
echo gef config gef.extra_plugins_dir gef-plugins >> ~/.gdbinit
```

## Additions

- Jemalloc heap manager: identify the heap memory regions, visualize heap allocations with metadata. Note that it is only tested on [CheriBSD jemalloc](https://github.com/CTSRD-CHERI/cheribsd/tree/main/contrib/jemalloc). 
    - `jheap chunk <address>`: inspect a heap allocation
    - `jheap chunks`: list in use heap allocations
- Scan for freed heap allocations that are pointed by valid capabilities in memory. Optionally, exclude capabilities stored in the heap.
    - `jheap uaf [noheap]`

## TODO

- More Jemalloc heap commands, like `arena(s)`, `extent(s)`, `tcache`, `slab(s)`
- List quarantined chunks, like `jheap quarantine chunks` (the quarantine is not tied to jemalloc manager right?)
- Snmalloc?
- PartitionAlloc?
