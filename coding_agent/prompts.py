def get_the_libraries_available():
    """
    Get the libraries available in the current environment.

    Returns:
        list: A list of library names.
    """
    import pkgutil

    return sorted([pkg.name for pkg in pkgutil.iter_modules()])


print(get_the_libraries_available())
