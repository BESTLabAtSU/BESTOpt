import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

class opt:
    def __init__(self, building_config):
        self.building = building_config
        self.globals = self.building.globals
        self.thermal_load = self.globals.Load_thermal if len(self.globals.Load_thermal) else None
        self.other_eleload = self.globals.Load_ele if len(self.globals.Load_ele) else 0
        self.tou = self.globals.TOU
        self.pv_size = self.building.pv.capacity_kw if self.building.pv else 0
        self.battery_size = self.building.battery.capacity_kwh if self.building.battery else 0
        self.ev_size = self.building.ev.capacity_kwh if self.building.ev else 0
        self.tes_size = self.building.tes.capacity_kwh if self.building.tes else 0
        self.operation_result = None

    def run(self, buildingname):
        # Run thermal optimization if TES exists
        if self.tes_size:
            self._tes_opt()
            total_eleload = self.other_eleload + self.tes_result["ele_opt"]
        else:
            total_eleload = self.other_eleload + self.tes_result["ele_base"]
        self.total_eleload = total_eleload
        # Run electric optimization 
        ele_df = self._ele_opt()
        self.buildingname = buildingname
        ele_df.to_csv(f'./opt_results/optimization{buildingname}.csv')

    def display(self, startday, endday):
        # Figure 1, TES vs no TES
        fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=300)
        ax.plot(np.arange(len(self.tes_result))[96*startday:96*endday], self.tes_result["ele_opt"][96*startday:96*endday], '-', linewidth=1, color='red', label="TES")
        ax.plot(np.arange(len(self.tes_result))[96*startday:96*endday], self.tes_result["ele_base"][96*startday:96*endday], '-', linewidth=1, color='black', label="no TES")
        ax_ = ax.twinx()
        ax_.plot(np.arange(len(self.tes_result))[96*startday:96*endday], self.tes_result["TOU"][96*startday:96*endday], '--', linewidth=1, color='gray', label="Price")
        ax.grid(False)
        ax_.grid(False)
        ax.set_ylabel('Chiller_Ele(kW)', fontsize=7)
        ax_.set_ylabel('Price($/kWh)', fontsize=7)
        ax.tick_params(axis='both', which='both', labelsize=7)
        ax_.tick_params(axis='both', which='both', labelsize=7)
        ax.set_xlabel('Time Step', fontsize=7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                  ncol=3, fontsize=7, frameon=False)
        plt.tight_layout()
        plt.show()

        # Figure 2, Baseline vs DERs
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.4), dpi=300)
        ax.plot(np.arange(len(self.ele_result))[96*startday:96*endday],
                self.ele_result["GridPurchase"][96*startday:96*endday], '-', linewidth=1, color='red', label="with DERs")
        ax.plot(np.arange(len(self.ele_result))[96*startday:96*endday], self.ele_result["Load"][96*startday:96*endday], '-', linewidth=1, color='black',
                label="no DERs")
        ax_ = ax.twinx()
        ax_.plot(np.arange(len(self.ele_result))[96*startday:96*endday], self.ele_result["TOU"][96*startday:96*endday], '--', linewidth=1, color='gray',
                 label="Price")
        ax.grid(False)
        ax_.grid(False)
        ax.set_ylabel('Total_Ele(kW)', fontsize=7)
        ax_.set_ylabel('Price($/kWh)', fontsize=7)
        ax.tick_params(axis='both', which='both', labelsize=7)
        ax_.tick_params(axis='both', which='both', labelsize=7)
        ax.set_xlabel('Time Step', fontsize=7)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                  ncol=3, fontsize=7, frameon=False)
        plt.tight_layout()
        plt.show()

    def evaluation(self, baseline=None, withmpc=None):
        tou = self.ele_result["TOU"].values
        grid = self.ele_result["GridPurchase"].values
        if baseline is None:
            baseline = self.ele_result["Load"].values
        else:
            cop = self.globals.cop
            baseline = self.other_eleload + baseline/cop
        tes = self.other_eleload + self.tes_result["ele_opt"]
        data = np.column_stack((tou, baseline, withmpc, tes, grid))
        df = pd.DataFrame(data, columns=['TOU Price', 'Baseline', 'Baseline_mpc', 'Baseline_mpc_tes', 'Baseline_mpc_tes_pv_bat'])
        df.to_csv(f'./opt_results/bill{self.buildingname}.csv')

        # Resolution and steps per day
        res_min = self.globals.Res
        steps_per_day = int(1440 / res_min)
        n_days = len(tou) // steps_per_day

        print("\n--- Daily Evaluation Summary ---")
        results = []
        for d in range(n_days):
            start = d * steps_per_day
            end = (d + 1) * steps_per_day

            tou_day = tou[start:end]
            grid_day = grid[start:end]
            base_day = baseline[start:end]

            # Dynamically identify off-peak and on-peak for the day
            unique_tou = np.sort(np.unique(tou_day))
            off_peak_rate = unique_tou[0]
            on_peak_rate = unique_tou[-1]
            off_mask = tou_day == off_peak_rate
            on_mask = tou_day == on_peak_rate

            # Compute metrics
            cost_with = np.sum(grid_day * tou_day)/4
            cost_base = np.sum(base_day * tou_day)/4
            cost_saving = cost_base - cost_with

            on_peak_reduction = base_day[on_mask].max() - grid_day[on_mask].max()
            on_peak_reduction_ratio = (base_day[on_mask].max() - grid_day[on_mask].max())/(base_day[on_mask].max())
            off_peak_increase = grid_day[off_mask].max() - base_day[off_mask].max()
            off_peak_increase_ratio = (grid_day[off_mask].max() - base_day[off_mask].max())/(grid_day[on_mask].max())
            on_peak_shifted = (base_day[on_mask].sum() - grid_day[on_mask].sum())/(base_day[on_mask].sum())
            results.append({
                "day": d + 1,
                "cost_saving": cost_saving,
                "on_peak_reduction": on_peak_reduction,
                "on_peak_reduction_ratio": on_peak_reduction_ratio,
                "off_peak_increase": off_peak_increase,
                "off_peak_increase_ratio": off_peak_increase_ratio,
                "on_peak_shifted": on_peak_shifted
            })

            # Optional: Print summary
            print(f"\nDay {d + 1}:")
            print(f"  Cost Saving: ${cost_saving:.2f}")
            print(f"  On-peak Load Reduction: {on_peak_reduction:.2f} kW, {on_peak_reduction_ratio * 100:.2f} %")
            print(f"  Off-peak Load Increase: {off_peak_increase:.2f} kW, {off_peak_increase_ratio * 100:.2f} %")
            print(f"  On-peak Energy Shifted: {on_peak_shifted * 100:.2f} %")

        # After the loop, convert to DataFrame
        df_summary = pd.DataFrame(results)
        df_summary.to_csv(f'./opt_results/evaluation{self.buildingname}.csv')

    def _tes_opt(self):
        N = len(self.thermal_load)
        tes_results = []
        SoC_tes_0 = self.building.tes.initial_soc if self.building.tes else 0
        one_day_step_size = int(1440 / self.globals.Res)
        tes = self.building.tes
        tes_max_chg = tes.c_rate * self.tes_size
        tes_max_dch = tes.c_rate * self.tes_size
        for i in range(0, N, one_day_step_size):  # Each optimization is based on one day horizon
            end_idx = min(i + one_day_step_size, N)
            window_steps = end_idx - i
            A = np.tril(np.ones((window_steps, window_steps)))  # State matrix

            m = gp.Model("Operation")
            m.setParam('OutputFlag', 0)
            m.setParam('Threads', self.globals.cpus)

            chiller_tes = m.addMVar((window_steps, 1), lb=0, name="Chiller2Tes")
            chiller_ahu = m.addMVar((window_steps, 1), lb=0, name="Chiller2Ahu")
            tes_ahu = m.addMVar((window_steps, 1), lb=0, name="Tes2Ahu")
            total_thermalload = self.thermal_load[i:end_idx].reshape(-1, 1)
            SoC_tes = m.addMVar((window_steps, 1), lb=0, name="SoC_Tes") if self.tes_size > 0 else None
            # air load balance constraint
            m.addConstr(chiller_ahu + tes_ahu == total_thermalload, name="Air_Load_Balance")
            # tes dynamic
            m.addConstr(
                SoC_tes_0 * np.ones((window_steps, 1)) +
                A @ (chiller_tes) * (self.globals.Res / 60) * tes.charge_efficiency / self.tes_size -
                A @ (tes_ahu) * (self.globals.Res / 60) / (self.tes_size * tes.discharge_efficiency) == SoC_tes,
                name="TES_Balance"
            )
            m.addConstr(SoC_tes >= tes.min_soc, name="Tes_SoCmin")
            m.addConstr(SoC_tes <= tes.max_soc, name="Tes_SoCmax")
            m.addConstr(chiller_tes <= tes_max_chg, name="Tes_ChargeLimit")
            m.addConstr(tes_ahu <= tes_max_dch, name="Tes_DisChargeLimit")
            tou = self.tou[i:end_idx].reshape(-1, 1)
            cop = self.globals.cop[i:end_idx].reshape(-1, 1)
            ele = (chiller_tes + chiller_ahu) / cop
            m.setObjective(tou.T @ ele, GRB.MINIMIZE)
            m.optimize()

            # Collect results
            result_data = {
                'Chiller2Tes': chiller_tes.X.reshape(-1, ),
                'Chiller2Ahu': chiller_tes.X.reshape(-1, ),
                'Tes2Ahu': tes_ahu.X.reshape(-1, ),
                'SoC_Tes': SoC_tes.X.reshape(-1, ),
                'TOU': tou.reshape(-1, ),
                'ele_opt': ((chiller_tes.X + chiller_ahu.X) / cop).reshape(-1, ),
                'ele_base': (total_thermalload / cop).reshape(-1, ),
            }

            result = pd.DataFrame(result_data, index=np.arange(i, end_idx))
            tes_results.append(result)

        self.tes_result = pd.concat(tes_results)
        return self.tes_result
        
    def _ele_opt(self):
        N = len(self.total_eleload)
        P_pv = self.building.pv.gen if self.building.pv else np.zeros_like(self.total_eleload)
        ele_results = []
        SoC_bat_0 = self.building.battery.initial_soc if self.building.battery else 0
        SoC_ev_0 = self.building.ev.arrival_soc if self.building.ev else 0
        one_day_step_size = int(1440 / self.globals.Res)

        for i in range(0, N, one_day_step_size):  # Each optimization is based on one day horizon
            end_idx = min(i + one_day_step_size, N)
            window_steps = end_idx - i
            A = np.tril(np.ones((window_steps, window_steps)))  # State matrix

            m = gp.Model("Operation")
            m.setParam('OutputFlag', 0)
            m.setParam('Threads', self.globals.cpus)

            # Load to Building
            PV_blgd = m.addMVar((window_steps, 1), lb=0, name="PV2Blgd")
            Grid_blgd = m.addMVar((window_steps, 1), lb=0, name="G2Blgd")
            Bat_blgd = m.addMVar((window_steps, 1), lb=0, name="Bat2Blgd")
            EV_blgd = m.addMVar((window_steps, 1), lb=0, name="EV2Blgd")

            # Load from PV
            PV_curtail = m.addMVar((window_steps, 1), lb=0, name="PVcurt")
            PV_bat = m.addMVar((window_steps, 1), lb=0, name="PV2Bat")
            PV_ev = m.addMVar((window_steps, 1), lb=0, name="PV2EV")

            # Load from Grid
            Grid_bat = m.addMVar((window_steps, 1), lb=0, name="G2Bat")
            Grid_ev = m.addMVar((window_steps, 1), lb=0, name="G2EV")
            Grid_purchase = m.addMVar((window_steps, 1), lb=0, name="GridPurchase")

            # Load between Battery and EV
            Bat_ev = m.addMVar((window_steps, 1), lb=0, name="Bat2EV")
            Ev_Bat = m.addMVar((window_steps, 1), lb=0, name="EV2Bat")

            # SOC of Battery and EV
            SoC_bat = m.addMVar((window_steps, 1), lb=0, name="SoC_Bat") if self.battery_size > 0 else None
            SoC_ev = m.addMVar((window_steps, 1), lb=0, name="SoC_EV") if self.ev_size > 0 else None

            total_load = self.total_eleload[i:end_idx].values.reshape(-1, 1)

            # Building load balance constraint
            m.addConstr(PV_blgd + Grid_blgd + Bat_blgd + EV_blgd == total_load, name="Bldg_Load_Balance")

            # PV generation balance constraint
            if self.pv_size > 0:
                m.addConstr(PV_blgd + PV_curtail + PV_bat + PV_ev == P_pv[i:end_idx].reshape(-1, 1),
                            name="PV_Gen_Balance")
            else:
                # If no PV, all PV-related variables should be zero
                m.addConstr(PV_blgd == 0, name="No_PV_to_Bldg")
                m.addConstr(PV_curtail == 0, name="No_PV_Curtail")
                m.addConstr(PV_bat == 0, name="No_PV_to_Bat")
                m.addConstr(PV_ev == 0, name="No_PV_to_EV")

            # Grid purchase balance constraint
            m.addConstr(Grid_blgd + Grid_bat + Grid_ev == Grid_purchase, name="Grid_Purchase_Balance")

            # Battery constraints
            if self.battery_size > 0:
                bat = self.building.battery
                bat_max_chg = bat.c_rate * self.battery_size
                bat_max_dch = bat.c_rate * self.battery_size
                m.addConstr(
                    SoC_bat_0 * np.ones((window_steps, 1)) +
                    A @ (Grid_bat + PV_bat + Ev_Bat) * (self.globals.Res/60) *bat.charge_efficiency / self.battery_size -
                    A @ (Bat_blgd + Bat_ev) * (self.globals.Res/60) / (self.battery_size * bat.discharge_efficiency) == SoC_bat,
                    name="Battery_Balance"
                )
                m.addConstr(SoC_bat >= bat.min_soc, name="Bat_SoCmin")
                m.addConstr(SoC_bat <= bat.max_soc, name="Bat_SoCmax")
                m.addConstr(Grid_bat + PV_bat + Ev_Bat <= bat_max_chg, name="Bat_ChargeLimit")
                m.addConstr(Bat_blgd + Bat_ev <= bat_max_dch, name="Bat_DisChargeLimit")
            else:
                # If no battery, all battery-related variables should be zero
                m.addConstr(Bat_blgd == 0, name="No_Bat_to_Bldg")
                m.addConstr(Grid_bat == 0, name="No_Grid_to_Bat")
                m.addConstr(PV_bat == 0, name="No_PV_to_Bat_constraint")
                m.addConstr(Bat_ev == 0, name="No_Bat_to_EV")
                m.addConstr(Ev_Bat == 0, name="No_EV_to_Bat")

            # EV constraints
            if self.ev_size > 0:
                ev = self.building.ev
                ev_max_chg = ev.c_rate * self.ev_size
                ev_max_dch = ev.c_rate * self.ev_size

                # Consider driving consumption
                driving_consumption = np.zeros((window_steps, 1))
                for t in range(window_steps):
                    global_t = i + t  # Global time index
                    daily_t = global_t % int(1440 / self.globals.Res)

                    # 1 hour after departure and 1 hour before arrival
                    departure_driving_end = (ev.departure_time + int(60 / self.globals.Res)) % int(
                        1440 / self.globals.Res)
                    arrival_driving_start = (ev.arrival_time - int(60 / self.globals.Res)) % int(
                        1440 / self.globals.Res)

                    # Check if in driving periods (Assume 1 hour before and after arrive and departure)
                    if ev.departure_time <= departure_driving_end:
                        is_driving = ev.departure_time <= daily_t < departure_driving_end
                    else:
                        is_driving = daily_t >= ev.departure_time or daily_t < departure_driving_end

                    if not is_driving:
                        if arrival_driving_start <= ev.arrival_time:
                            is_driving = arrival_driving_start <= daily_t < ev.arrival_time
                        else:
                            is_driving = daily_t >= arrival_driving_start or daily_t < ev.arrival_time

                    if is_driving:
                        driving_consumption[t, 0] = 0.05 * (self.globals.Res / 60)  # 5% per hour

                # Modified EV balance constraint with driving consumption
                m.addConstr(
                    SoC_ev_0 * np.ones((window_steps, 1)) +
                    A @ (Grid_ev + PV_ev + Bat_ev) * (self.globals.Res / 60) * ev.charge_efficiency / self.ev_size -
                    A @ (EV_blgd + Ev_Bat) * (self.globals.Res / 60) / (self.ev_size * ev.discharge_efficiency) -
                    A @ driving_consumption == SoC_ev,  # Add driving consumption here
                    name="EV_Balance"
                )

                m.addConstr(SoC_ev >= ev.min_soc, name="EV_SoCmin")
                m.addConstr(SoC_ev <= ev.max_soc, name="EV_SoCmax")
                m.addConstr(Grid_ev + PV_ev + Bat_ev <= ev_max_chg, name="EV_ChargeLimit")
                m.addConstr(EV_blgd + Ev_Bat <= ev_max_dch, name="EV_DisChargeLimit")

                # EV availability constraints
                for t in range(window_steps):
                    if t == ev.departure_time - 1:
                        m.addConstr(SoC_ev[t, 0] >= ev.required_departure_soc, name=f"EV_Required_SOC_at_departure")
                    if t < ev.arrival_time and t >= ev.departure_time:
                        m.addConstr(EV_blgd[t, 0] == 0, name="No_EV_to_Bldg")
                        m.addConstr(Grid_ev[t, 0] == 0, name="No_Grid_to_EV")
                        m.addConstr(PV_ev[t, 0] == 0, name="No_PV_to_EV_constraint")
                        m.addConstr(Bat_ev[t, 0] == 0, name="No_Bat_to_EV_constraint")
                        m.addConstr(Ev_Bat[t, 0] == 0, name="No_EV_to_Bat_constraint")
            else:
                # If no EV, all EV-related variables should be zero
                m.addConstr(EV_blgd == 0, name="No_EV_to_Bldg")
                m.addConstr(Grid_ev == 0, name="No_Grid_to_EV")
                m.addConstr(PV_ev == 0, name="No_PV_to_EV_constraint")
                m.addConstr(Bat_ev == 0, name="No_Bat_to_EV_constraint")
                m.addConstr(Ev_Bat == 0, name="No_EV_to_Bat_constraint")

            tou = self.tou[i:end_idx].reshape(-1, 1)
            m.setObjective(tou.T @ Grid_purchase, GRB.MINIMIZE)
            m.optimize()

            # Collect results
            result_data = {
                'PV2Blgd': PV_blgd.X.reshape(-1, ),
                'G2Blgd': Grid_blgd.X.reshape(-1, ),
                'Bat2Blgd': Bat_blgd.X.reshape(-1, ),
                'EV2Blgd': EV_blgd.X.reshape(-1, ),
                'GridPurchase': Grid_purchase.X.reshape(-1, ),
                'PVcurt': PV_curtail.X.reshape(-1, ),
                'PV2Bat': PV_bat.X.reshape(-1, ),
                'PV2EV': PV_ev.X.reshape(-1, ),
                'G2Bat': Grid_bat.X.reshape(-1, ),
                'G2EV': Grid_ev.X.reshape(-1, ),
                'Bat2EV': Bat_ev.X.reshape(-1, ),
                'EV2Bat': Ev_Bat.X.reshape(-1, ),
                'TOU': tou.reshape(-1, ),
                'Load': total_load.reshape(-1, ),
            }

            # SOC variables need special handling since they might be None
            if SoC_bat is not None:
                result_data['SoC_Bat'] = SoC_bat.X.reshape(-1, )
            else:
                result_data['SoC_Bat'] = np.zeros(window_steps)

            if SoC_ev is not None:
                result_data['SoC_EV'] = SoC_ev.X.reshape(-1, )
            else:
                result_data['SoC_EV'] = np.zeros(window_steps)

            result = pd.DataFrame(result_data, index=np.arange(i, end_idx))
            ele_results.append(result)

        self.ele_result = pd.concat(ele_results)
        return self.ele_result
