# GEF plugins

## Setup

Prerequisite: [gef with CHERI support](https://github.com/CTSRD-CHERI/gef)

```sh
# assuming this directory is named gef-plugins
echo gef config gef.extra_plugins_dir gef-plugins >> ~/.gdbinit
```

## Additions

- Jemalloc heap manager: identify the heap memory regions, visualize heap chunks with metadata. Note that it is only tested on [CheriBSD jemalloc](https://github.com/CTSRD-CHERI/cheribsd/tree/main/contrib/jemalloc). 
    - `jheap chunk <address>`: inspect a heap chunk
    - `jheap chunks`: list in use heap chunks
- Scan for freed heap chunks that are pointed by valid capabilities in memory. Optionally, exclude capabilities stored in the heap.
    - `jheap uaf [noheap]`

## TODO

- More Jemalloc heap commands, like `bins`, `arenas`.
- Snmalloc?
