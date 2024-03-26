
@register
class MrsCommand(GenericCommand):
    """Base command to get information about heap caprevocation state."""

    _cmdline_ = "mrs"
    _syntax_  = f"{_cmdline_} (info|chunk|quarantine)"

    def __init__(self) -> None:
        super().__init__(prefix=True)
        return

    @only_if_gdb_running
    def do_invoke(self, _: List[str]) -> None:
        self.usage()
        return
    

@register
class MrsInfoCommand(GenericCommand):
    """Display information about the mrs quarantine, global state and
    the revocation bitmap."""

    _cmdline_ = "mrs info"
    _syntax_  = f"{_cmdline_} [-h] [all] [thread_ids...]"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        current_thread = gdb.selected_thread()
        if current_thread is None:
            err("Couldn't find current thread")
            return

        # As a nicety, we want to display threads in ascending order by gdb number
        threads = sorted(gdb.selected_inferior().threads(), key=lambda t: t.num)
        if argv:
            if "all" in argv:
                tids = [t.num for t in threads]
            else:
                tids = self.check_thread_ids([int(a) for a in argv])
        else:
            tids = [current_thread.num]

        # loop through all selected threads (default current thread)
        for thread in threads:
            if thread.num not in tids:
                continue
            thread.switch()

            gef_print(titlify(f"Thread {thread.num:d}"))

            gef_print(f"Allocated size: {gef.heap_caprevoke.allocated_size:#x}")
            gef_print(f"Max allocated size: {gef.heap_caprevoke.max_allocated_size:#x}")

            gef_print(f"Quarantine size: {gef.heap_caprevoke.quarantine_size:#x}")
            gef_print(f"Quarantine max size: {gef.heap_caprevoke.quarantine_max_size:#x}")

            entire_revocation_bitmap = gef.heap_caprevoke.entire_shadow
            gef_print(f"Entire revocation map capability: {str(entire_revocation_bitmap)}")

            # XXXR3: epochs? cheri_revoke_info?

        current_thread.switch()


@register
class MrsQuarantineCommand(GenericCommand):
    """Display information about the mrs quarantine, global state and
    the revocation bitmap."""

    _cmdline_ = "mrs quarantine"
    _syntax_  = f"{_cmdline_} [-h]"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        # display quarantined chunks as capabilities as they are stored
        quarantined_chunks = gef.heap_caprevoke.chunks
        chunk_sizes = gef.heap_caprevoke.chunk_sizes
        gef_print(f"application_quarantine[size={gef.heap_caprevoke.quarantine_size:#x}, count={len(quarantined_chunks)}] ")
        for chunk, chunk_size in zip(quarantined_chunks, chunk_sizes):
            gef_print(f" {RIGHT_ARROW} {Color.colorify('Chunk', 'yellow bold underline')}(addr={chunk:#x}, size={chunk_size:#x})")


@register
class MrsChunkCommand(GenericCommand):
    """Display information about the mrs quarantine, global state and
    the revocation bitmap."""

    _cmdline_ = "mrs chunk"
    _syntax_  = f"{_cmdline_} [-h] address"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @parse_arguments({"address": ""}, {})
    @only_if_gdb_running
    def do_invoke(self, _: List[str], **kwargs: Any) -> None:
        args = kwargs["arguments"]
        if not args.address:
            err("Missing heap address")
            self.usage()
            return

        addr = parse_address(args.address)

        # XXXR3: use this command with the underlying allocation address or this breaks.
        # But this is allocator specific... So query jemalloc heap manager or snmalloc heap
        # manager first. 

        revocation_bit_addr = gef.heap_caprevoke.get_shadow_bit_addr(addr)
        revocation_bit = gef.heap_caprevoke.get_shadow_bit(addr)
        revocation_status = Color.colorify('Yes', 'green') if revocation_bit == 1 else Color.colorify('No', 'red')

        quarantined_chunks = gef.heap_caprevoke.chunks
        quarantine_status = Color.colorify('Yes', 'green') if addr in quarantined_chunks else Color.colorify('No', 'red')

        gef_print(f"{Color.colorify('Chunk', 'yellow bold underline')}(addr={addr:#x})")
        gef_print(f"In quarantine: {quarantine_status}") # quarantined chunk is the underlying allocation
        gef_print(f"Revocation bit address: {revocation_bit_addr:#x}")
        gef_print(f"Revocation bit set (first word): {revocation_status}")
