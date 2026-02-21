import pandapower as pp
import pandapower as pp
import pandapower.networks as pn
import pandapower.networks as nw # Alias used in my mental model, mapping to pn actually

try:
    import simbench as sb
except ImportError:
    sb = None

class GridBuilder:
    """
    A class to generate various grid topologies using pandapower.
    """

    @staticmethod
    def create_ieee13_grid():
        """
        Creates the IEEE 13-bus test case manually as in the original notebook 
        or using pandapower networks if available (using manual construction for fidelity to original).
        """
        net = pp.create_empty_network()

        # Create Buses (Nodes) with Geodata for Visualization
        b_source = pp.create_bus(net, vn_kv=115, name="Source", geodata=(0, 10))
        b650 = pp.create_bus(net, vn_kv=4.16, name="650", geodata=(0, 8))
        b632 = pp.create_bus(net, vn_kv=4.16, name="632", geodata=(0, 6))
        b671 = pp.create_bus(net, vn_kv=4.16, name="671", geodata=(0, 4)) 
        b680 = pp.create_bus(net, vn_kv=4.16, name="680", geodata=(-2, 2))
        b684 = pp.create_bus(net, vn_kv=4.16, name="684", geodata=(2, 2))
        b611 = pp.create_bus(net, vn_kv=4.16, name="611", geodata=(2, 0))
        b652 = pp.create_bus(net, vn_kv=4.16, name="652", geodata=(4, 2))


        # External Grid Connection
        pp.create_ext_grid(net, bus=b_source, vm_pu=1.05)

        # Transformer (115kV -> 4.16kV)
        pp.create_transformer_from_parameters(net, hv_bus=b_source, lv_bus=b650,
                                              sn_mva=5, vn_hv_kv=115, vn_lv_kv=4.16,
                                              vkr_percent=1, vk_percent=8,
                                              pfe_kw=0, i0_percent=0, name="Substation")

        # Lines (Cables)
        pp.create_line_from_parameters(net, b650, b632, length_km=0.6, r_ohm_per_km=0.3, x_ohm_per_km=0.2, c_nf_per_km=10, max_i_ka=0.4, name="Line_650_632")
        pp.create_line_from_parameters(net, b632, b671, length_km=0.6, r_ohm_per_km=0.3, x_ohm_per_km=0.2, c_nf_per_km=10, max_i_ka=0.4, name="Line_632_671")
        pp.create_line_from_parameters(net, b671, b680, length_km=0.3, r_ohm_per_km=0.4, x_ohm_per_km=0.3, c_nf_per_km=10, max_i_ka=0.4, name="Line_671_680")
        pp.create_line_from_parameters(net, b671, b684, length_km=0.2, r_ohm_per_km=0.4, x_ohm_per_km=0.3, c_nf_per_km=10, max_i_ka=0.4, name="Line_671_684")
        pp.create_line_from_parameters(net, b684, b611, length_km=0.1, r_ohm_per_km=0.6, x_ohm_per_km=0.5, c_nf_per_km=10, max_i_ka=0.4, name="Line_684_611")
        pp.create_line_from_parameters(net, b684, b652, length_km=0.2, r_ohm_per_km=0.6, x_ohm_per_km=0.5, c_nf_per_km=10, max_i_ka=0.4, name="Line_684_652")

        # Loads
        pp.create_load(net, bus=b671, p_mw=1.155, q_mvar=0.660, name="Load_671")
        pp.create_load(net, bus=b632, p_mw=0.200, q_mvar=0.100, name="Load_632")
        pp.create_load(net, bus=b680, p_mw=0.400, q_mvar=0.200, name="Load_680")
        pp.create_load(net, bus=b611, p_mw=0.170, q_mvar=0.080, name="Load_611")
        pp.create_load(net, bus=b652, p_mw=0.128, q_mvar=0.086, name="Load_652")

        # Capacitor Bank
        pp.create_sgen(net, bus=b671, p_mw=0, q_mvar=0.600, name="Capacitor_675")

        return net

    @staticmethod
    def create_ieee33_grid():
        """Returns the IEEE 33-bus radial distribution system."""
        return pn.case33bw()

    @staticmethod
    def create_ieee123_grid():
        """Returns the IEEE 123-node test feeder."""
        return pn.case123()

    @staticmethod
    def create_simbench_grid(grid_code="1-LV-rural1--0-sw"):
        """
        Returns a SimBench grid. 
        Requires 'simbench' package installed.
        """
        if sb is None:
            raise ImportError("SimBench package is not installed. Please install it using 'pip install simbench'.")
        
        return sb.get_simbench_net(grid_code)

    @staticmethod
    def get_grid(grid_name):
        """Factory method to get grid by name."""
        grid_name = grid_name.lower()
        if grid_name == "ieee13":
            return GridBuilder.create_ieee13_grid()
        elif grid_name == "ieee33":
            return GridBuilder.create_ieee33_grid()
        elif grid_name == "ieee123":
            return GridBuilder.create_ieee123_grid()
        elif grid_name == "ieee39":
            return pn.case39()
        elif grid_name == "ieee118":
            return pn.case118()
        elif grid_name == "ieee300":
            return pn.case300()
        elif grid_name == "cigre_mv":
            return pn.create_cigre_network_mv(with_der="all")
        elif grid_name == "gbnetwork":
            return pn.GBnetwork()
        elif grid_name == "simbench":
            return GridBuilder.create_simbench_grid()
        else:
            raise ValueError(f"Unknown grid name: {grid_name}")

    @staticmethod
    def available_models():
        return ["ieee13", "ieee33", "ieee39", "ieee123", "ieee118", "ieee300", "cigre_mv", "gbnetwork", "simbench"]

