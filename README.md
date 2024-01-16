# GEF plugins

## Setup

```sh
# assuming this directory is named gef-plugins
echo gef config gef.extra_plugins_dir gef-plugins >> ~/.gdbinit
```

## Additions

- Jemalloc heap manager: identify the heap memory regions, visualize heap chunks with metadata. Note that it is only tested on [CheriBSD jemalloc](https://github.com/CTSRD-CHERI/cheribsd/tree/main/contrib/jemalloc). 
    - `jheap chunk <address>`
- Scan for freed heap chunks that are pointed by valid capabilities in memory. Optionally, exclude capabilities stored in the heap.
    - `jheap uaf [noheap]`
- Scan memory for valid capabilities stored in `source` and pointing to `destination`. 
    - `scancaps [source] [destination]` 
