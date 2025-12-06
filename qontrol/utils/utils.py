from dynamiqs.time_qarray import ConstantTimeQArray, SummedTimeQArray, TimeQArray


def get_hamiltonians(H: TimeQArray) -> list:
    """Get a list of all Hamiltonian terms that have a prefactor method.

    We use this method both for plotting the controls and for calculating cost functions
    with respect to the values of the controls. In both cases, we are only interested in
    Hamiltonian terms that have prefactor methods, so we need to recursively plumb down
    through all of the SummedTimeQArray instances and ignore any ConstantTimeQArray.
    """
    if not isinstance(H, ConstantTimeQArray | SummedTimeQArray):
        return [H]
    if isinstance(H, SummedTimeQArray):
        Hs = []
        for _H in H.timeqarrays:
            Hs.extend(get_hamiltonians(_H))
        return Hs
    # ConstantTimeQArray, can't plot or ask for prefactor
    return []
