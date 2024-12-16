from functools import cached_property

import numpy as np

g = 9.81


class ChannelProperties:
    """
    A class for the properties of a channel.
    Attributes
    ----------
    width : float
        The width of the channel. (w)
    height : float
        The (tidally averaged) dept of the channel. (h)
    length : float
        The length of the channel.
    tidal_averaged_flow : float
        The tidal averaged flow in the channel. (phi_0)
    drag_coefficient : float
        The drag coefficient of the bottom of the channel. (c_d)
    tidal_velocity_amplitude : float
        A characteristic velocity for the tidal flow in the channel.
        This is used to calculate the friction factor and the effective diffusion coefficient. (U)
    diffusion_prefactor : float, optional
        The diffusion prefactor, default is 0.0125. Must be between 0.005 and 0.020.

    Methods
    -------
    frictionless_wave_velocity:
        Calculate the wave velocity of the eta/phi wave without friction.
    friction_factor:
        Calculate the friction factor. (r)
    effective_diffusion_coefficient:
        Calculate the effective diffusion coefficient (𝔻), needs a diffusion prefactor between 0.005 and 0.020 that depends on the channel's empirical data.
    __repr__:
        Return a string representation of the ChannelProperties instance.
    """

    def __init__(
        self,
        width: float,
        height: float,
        length: float,
        tidal_averaged_flow: float,
        drag_coefficient: float,
        tidal_velocity_amplitude: float,
        diffusion_prefactor: float = 0.0125,
    ):
        self.width: float = width
        self.height: float = height
        self.length: float = length
        self.tidal_averaged_flow: float = tidal_averaged_flow
        self.drag_coefficient: float = drag_coefficient
        self.tidal_velocity_amplitude: float = tidal_velocity_amplitude

        if diffusion_prefactor < 0.005 or diffusion_prefactor > 0.020:
            raise ValueError("Diffusion prefactor out of range")
        self.diffusion_prefactor: float = diffusion_prefactor

    @cached_property
    def frictionless_wave_velocity(self) -> float:
        """Calculate the wave velocity of the eta/phi wave withou friction."""
        return np.sqrt(g * self.height)

    @cached_property
    def friction_factor(self) -> float:
        """Calculate the friction factor, assumes the tidal velocity amplitude is a constant"""
        # ToDo: Een check invoeren voor de tidal velocity amplitude?
        return (self.drag_coefficient / self.height) * (
            8 * self.tidal_velocity_amplitude / (3 * np.pi)
        )

    @cached_property
    def effective_diffusion_coefficient(self) -> float:
        """Calculate the effective diffusion coefficient, needs a diffusion prefactor
        between 0.005 and 0.020 that depends on the channels empirical data"""
        return (
            self.diffusion_prefactor
            * ((self.width**2) * self.tidal_velocity_amplitude)
            / (np.sqrt(self.drag_coefficient) * self.height)
        )

    def __repr__(self) -> str:
        return (
            f"ChannelProperties("
            f"width={self.width}, "
            f"height={self.height}, "
            f"length={self.length}, "
            f"tidal_averaged_flow={self.tidal_averaged_flow}, "
            f"drag_coefficient={self.drag_coefficient}, "
            f"tidal_velocity_amplitude={self.tidal_velocity_amplitude}, "
            f"diffusion_prefactor={self.diffusion_prefactor}"
            f")"
        )
