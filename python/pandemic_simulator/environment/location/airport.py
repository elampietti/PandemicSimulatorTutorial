from dataclasses import dataclass

from ..interfaces import NonEssentialBusinessLocationState, ContactRate, SimTimeTuple, NonEssentialBusinessBaseLocation

__all__ = ['Airport', 'AirportState']


@dataclass
class AirportState(NonEssentialBusinessLocationState):
    # TODO: Decide on these contact rates
    contact_rate: ContactRate = ContactRate(4, 2, 5, 0.1, 0.1, 0.1)
    # Airports are open 24/7
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(0, 24)), week_days=tuple(range(0, 7)))


class Airport(NonEssentialBusinessBaseLocation[AirportState]):
    """Implements an airport"""

    state_type = AirportState