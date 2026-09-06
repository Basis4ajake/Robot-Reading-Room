import threading

from local_knowledge_library.api.state import AppState, LibraryBusyError


def _make_state(tmp_path):
    state = AppState(data_dir=str(tmp_path), force_dummy=True)
    state.registry.create_library(library_id="lib-a", name="Lib A")
    state.registry.create_library(library_id="lib-b", name="Lib B")
    return state


def test_exclusive_raises_busy_while_use_runtime_is_held(tmp_path):
    state = _make_state(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def hold_runtime():
        with state.use_runtime("lib-a"):
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=hold_runtime)
    thread.start()
    try:
        assert entered.wait(timeout=5), "use_runtime never entered"
        try:
            with state.exclusive("lib-a"):
                assert False, "expected LibraryBusyError while use_runtime is held"
        except LibraryBusyError:
            pass
    finally:
        release.set()
        thread.join(timeout=5)

    # Busy only while actually held - released cleanly afterward.
    with state.exclusive("lib-a"):
        pass


def test_use_runtime_raises_busy_while_exclusive_is_held(tmp_path):
    state = _make_state(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def hold_exclusive():
        with state.exclusive("lib-a"):
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=hold_exclusive)
    thread.start()
    try:
        assert entered.wait(timeout=5), "exclusive never entered"
        try:
            with state.use_runtime("lib-a"):
                assert False, "expected LibraryBusyError while exclusive is held"
        except LibraryBusyError:
            pass
    finally:
        release.set()
        thread.join(timeout=5)

    with state.use_runtime("lib-a"):
        pass


def test_different_libraries_do_not_block_each_other(tmp_path):
    state = _make_state(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def hold_runtime_a():
        with state.use_runtime("lib-a"):
            entered.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=hold_runtime_a)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        # lib-b is unrelated - must not be affected by lib-a being busy.
        with state.exclusive("lib-b"):
            pass
        with state.use_runtime("lib-b"):
            pass
    finally:
        release.set()
        thread.join(timeout=5)


def test_exclusive_does_not_leave_library_stuck_busy_after_an_error(tmp_path):
    """A config update that raises inside the exclusive() block (e.g. the
    library doesn't exist) must still release the lock - otherwise every
    future request against that library_id would wrongly see it as busy
    forever."""
    state = _make_state(tmp_path)

    try:
        with state.exclusive("lib-a"):
            raise ValueError("simulated failure mid-update")
    except ValueError:
        pass

    with state.use_runtime("lib-a"):
        pass


def test_invalidate_closes_and_drops_the_cached_runtime(tmp_path):
    state = _make_state(tmp_path)
    with state.use_runtime("lib-a") as runtime:
        cached_store = runtime.vector_store

    state.invalidate("lib-a")
    assert "lib-a" not in state._runtimes

    with state.use_runtime("lib-a") as runtime2:
        assert runtime2.vector_store is not cached_store


def test_close_force_closes_everything_regardless_of_tracked_use(tmp_path):
    state = _make_state(tmp_path)
    with state.use_runtime("lib-a"):
        pass
    with state.use_runtime("lib-b"):
        pass

    state.close()

    assert state._runtimes == {}
    assert state._active_uses == {}
