def import_pyvista():
    try:
        import pyvista as pv

        return pv
    except ImportError:
        raise Exception("You need to install pyvista to use this function")
