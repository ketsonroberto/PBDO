import numpy as np
import copy
import sys


# from BuildingProperties import *


class Structure:

    def __init__(self, building=None, columns=None, slabs=None, core=None, concrete=None, steel=None, cost=None):
        self.building = building
        self.columns = columns
        self.slabs = slabs
        self.core = core
        self.concrete = concrete
        self.steel = steel
        self.cost = cost

    def stiffness_story(self):
        area_col = self.columns["area"]
        moment_inertia_col = self.columns["Iy"]
        height_col = self.columns["height"]
        k_col = self.stiffness(area=area_col, moment_inertia=moment_inertia_col, height=height_col)

        area_core = self.core["area"]
        moment_inertia_core = self.core["Iy"]
        height_core = self.core["height"]
        k_core = self.stiffness(area=area_core, moment_inertia=moment_inertia_core, height=height_core)

        num_col = self.columns["quantity"]
        num_core = self.core["quantity"]

        k_story = num_col * k_col + num_core * k_core

        return k_story

    def stiffness(self, area=None, moment_inertia=None, height=None):
        Gc = self.concrete["Gc"]
        Ec = self.concrete["Ec"]
        Es = self.steel["Es"]
        As = self.columns["v_steel"] * area
        Ac = area - As
        E = (As * Es + Ac * Ec) / (area)

        ks = Gc * area / height
        kf = 3 * E * moment_inertia / (height ** 3)

        kt = 1 / ((1 / kf) + (1 / ks))

        return kt

    def mass_storey(self, top_story=False):

        num_col = self.columns["quantity"]
        mslab = self.mass_slab()
        mcol = self.mass_column()

        if top_story:
            mass_st = 0.5 * (num_col * mcol) + mslab
        else:
            mass_st = num_col * mcol + mslab

        return mass_st

    def mass_slab(self):

        ros = self.steel["density"]
        ro = self.concrete["density"]
        thickness = self.slabs["thickness"]
        width = self.slabs["width"]
        depth = self.slabs["depth"]

        Vs = self.slabs["steel_rate"] * thickness * width * depth
        Vc = thickness * width * depth - Vs

        mass_s = ro * Vc + ros * Vs

        return mass_s

    def mass_column(self):

        ros = self.steel["density"]
        ro = self.concrete["density"]
        height = self.columns["height"]
        area = self.columns["area"]

        As = self.columns["v_steel"] * area
        Ac = area - As

        mass_col = ro * Ac * height + ros * As * height  # +stirups

        return mass_col

    def compression(self, col_size=None, L=None):

        # For a single column
        PA_ksi = 1.4503773800722e-7
        fc_prime = self.concrete["fck"]
        Ec = self.concrete["Ec"]
        fy = self.steel["fy"]
        Es = self.steel["Es"]
        fc_prime = fc_prime * PA_ksi
        ecu = 0.003
        fy = fy * PA_ksi
        Es = Es * PA_ksi
        Ec = Ec * PA_ksi

        ros = 0.08

        b = col_size
        b = b / 0.0254  # m to inch
        h = copy.copy(b)
        As = ros*b*h/2
        As_prime = ros*b*h/2
        d_prime = 2.5
        d = b - d_prime

        # Centroid
        yc = h/2
        n = Es / Ec
        y_bar = (b * h * yc + (n - 1) * As_prime * d_prime + (n - 1) * As * d) / (
                    b * h + (n - 1) * As_prime + (n - 1) * As)

        # uncracked moment of inertia.
        Iun = (b * h ** 3 / 12) + b * h * (y_bar - yc) ** 2 + (n - 1) * As_prime * (y_bar - d_prime) ** 2 +\
              (n - 1) * As * (d - y_bar) ** 2

        # tensile strenght concrete
        ft = 7.5 * np.sqrt(1000 * fc_prime) / 1000

        # yb is the distance of y_bar to the bottom of the section
        yd = h - y_bar
        Mcr = ft * Iun / yd
        phi_cr = Mcr / (Ec * Iun) # rad / in

        M = np.zeros(4)
        phi = np.zeros(4)
        M[0] = 0
        phi[0] = 0
        M[1] = Mcr
        phi[1] = phi_cr

        # 2 - cracked transformed section.
        # kd is the height of the compression section of the cracked section

        # (b/2)*kd^2 + ((n-1)*As_prime + n*As)*kd - ((n-1)*As_prime*d_prime + n*As*d);

        aa = b / 2
        bb = (n - 1) * As_prime + n * As
        cc = - ((n - 1) * As_prime * d_prime + n * As * d)

        #depth of the neutral axis from the top of the section.
        kd = (-bb + np.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)

        Icr = (b * kd ** 3 / 12) + b * kd * (kd / 2) ** 2 + (n - 1) * As_prime * (kd - d_prime) ** 2 + (n) * As * (
                    d - kd) ** 2
        phi_acr = Mcr / (Ec * Icr)
        #M[2] = Mcr
        #phi[2] = phi_acr

        # 3 -  Yield of steel or concrete non-linear
        # cracked transformed section valid until fs=fy or fc = 0.7fc_prime
        # steel yields es = ey = fy/Es

        if d<=kd:
            # No yielding
            phi_y = sys.float_info.max
            My = sys.float_info.max
        else:
            es = fy / Es
            phi_y = es / (d - kd)
            My = phi_y * Ec * Icr

        # concrete nonlinear.
        phi_con = 0.7 * (fc_prime / Ec) / kd
        Mcon = phi_con * Ec * Icr #kip-in

        # check which one occur first (yielding or nonlinear concrete)
        if My < Mcon:
            Mnl = My
            phi_nl = phi_y
        else:
            Mnl = Mcon
            phi_nl = phi_con

        M[2] = Mnl
        phi[2] = phi_nl

        # Find nominal strength (ACI) from strain-stress diagram.
        if fc_prime <= 4:
            b1 = 0.85
        elif 4 < fc_prime <= 8:
            b1 = 0.85 - 0.05 * (fc_prime - 4)
        else:
            b1 = 0.65

        # Find c and fs_prime
        cont = 1
        c = 1
        ct = 0.01
        fs_prime = copy.copy(fy)
        while abs(c / ct - 1) > 0.0002 and cont < 100:
            c = (As * fy - As_prime * fs_prime) / (0.85 * fc_prime * b * b1)
            if c==0:
                c=0.00000000001
            c=abs(c)

            fs_prime = 0.003 * Es * ((c - d_prime) / c)
            cont = cont + 1

        phi_r = ecu / c
        As2 = As_prime * fs_prime / fy
        Mr = As2 * fy * (d - d_prime) + (As - As2) * fy * (d - b1 * c / 2)
        M[3] = Mr
        phi[3] = phi_r

        return M, phi

    def deformation_damage_index(self, B=None, stiffness=None, Mom=None, phi=None):

        k = stiffness

        Lc = self.columns["height"]
        EI = (k * Lc ** 3) / 12
        M = 6 * EI * B / (Lc ** 2)
        NM_kipin = 112.9848004306
        M = M/NM_kipin
        My = Mom[1]
        phiy = phi[1]
        Mu = Mom[2]
        phiu = phi[2]
        phim = phiu

        if M <= Mom[1]:
            phim = M * phi[1] / Mom[1]
        elif Mom[1] < M <= Mom[2]:
            phim = ((M - Mom[1]) / (Mom[2] - Mom[1])) * (phi[2] - phi[1]) + phi[1]
        elif Mom[2] < M <= Mom[3]:
            phim = ((M - Mom[2]) / (Mom[3] - Mom[2])) * (phi[3] - phi[2]) + phi[2]
        else:
            phim = phi[3]

        if phim < phi[1]:
            ddi = 0
        elif phi[1] <= phim < phi[2]:
            ddi = (phim - phi[1]) / (phi[2] - phi[1])
        elif phi[2] <= phim < phi[3]:
            ddi = (phim - phi[2]) / (phi[3] - phi[2])
        else:
            ddi = 1.0

        return ddi


class Costs(Structure):

    def __init__(self, building=None, columns=None, slabs=None, core=None, concrete=None, steel=None, cost=None):
        self.building = building
        self.columns = columns
        self.slabs = slabs
        self.core = core
        self.concrete = concrete
        self.steel = steel
        self.cost = cost

        Structure.__init__(self, building=building, columns=columns, slabs=slabs, core=core, concrete=concrete,
                           steel=steel, cost=cost)

    def initial_cost_stiffness(self, col_size=None, par0=None, par1=None):
        
        num_col = self.columns["quantity"]
        height_col = self.columns["height"]
        pslabs = self.slabs["cost_m2"]
        
        area_col = col_size**2
        moment_inertia_col = (col_size**4)/12
        k_col = self.stiffness(area=area_col, moment_inertia=moment_inertia_col, height=height_col)
        
        stiffness_kN_cm = 0.00001 * k_col
        cost_initial = (par0 * (stiffness_kN_cm) ** par1) * num_col * height_col
        cost_initial = cost_initial + pslabs * self.slabs["width"] * self.slabs["depth"]  # price_slabs_m2*A

        cost_initial = 1.6*cost_initial # include 60% of additional costs.
        return cost_initial

    def cost_damage(self, b=None, col_size=None, L=None, ncolumns=None, dry_wall_area=None):

        A_glazing = 1.5 * L
        A_bulding = 2 * L * (self.building["width"] + self.building["depth"])
        Adry = 5.95

        IDRd = self.cost["IDRd"]
        IDRu = self.cost["IDRu"]
        cIDRd = self.cost["cost_IDRd"]
        cIDRu = self.cost["cost_IDRu"]

        IDRd_eg = self.cost["IDRd_eg"]
        IDRu_eg = self.cost["IDRu_eg"]
        cIDRd_eg = self.cost["cost_IDRd_eg"]
        cIDRu_eg = self.cost["cost_IDRu_eg"]

        IDRd_dp = self.cost["IDRd_dp"]
        IDRu_dp = self.cost["IDRu_dp"]
        cIDRd_dp = self.cost["cost_IDRd_dp"]
        cIDRu_dp = self.cost["cost_IDRu_dp"]

        IDRd_df = self.cost["IDRd_df"]
        IDRu_df = self.cost["IDRu_df"]
        cIDRd_df = self.cost["cost_IDRd_df"]
        cIDRu_df = self.cost["cost_IDRu_df"]

        # COLUMNS - SLAB CONECTIONS
        bsf = IDRd * L
        bcol = IDRu * L
        csf = ncolumns * cIDRd
        ccol = ncolumns * cIDRu
        # bar(1) = datad % cost_par % bcol(i)

        # EXTERIOR GLAZING
        bsf_eg = IDRd_eg * L
        bcol_eg = IDRu_eg * L
        csf_eg = cIDRd_eg * (A_bulding / A_glazing)
        ccol_eg = cIDRu_eg * (A_bulding / A_glazing)
        # bar(2) = datad % cost_par % bcol_eg(i)

        # DRYWALL PARTITIONS
        bsf_dp = IDRd_dp * L
        bcol_dp = IDRu_dp * L
        csf_dp = cIDRd_dp * (dry_wall_area / Adry)
        ccol_dp = cIDRu_dp * (dry_wall_area / Adry)
        # bar(3) = datad % cost_par % bcol_dp(i)

        # DRYWALL FINISH
        bsf_df = IDRd_df * L
        bcol_df = IDRu_df * L
        csf_df = cIDRd_df * (dry_wall_area / Adry)
        ccol_df = cIDRu_df * (dry_wall_area / Adry)
        # bar(4) = datad % cost_par % bcol_df(i)

        if b < bsf:
            cf_cs = 0
        elif bcol > b >= bsf:
            cf_cs = ((ccol - csf) / (bcol - bsf)) * (b - bsf) + csf
        else:
            cf_cs = ccol

        if b < bsf_eg:
            cf_eg = 0
        elif bcol_eg > b >= bsf_eg:
            cf_eg = ((ccol_eg - csf_eg) / (bcol_eg - bsf_eg)) * (b - bsf_eg) + csf_eg
        else:
            cf_eg = ccol_eg

        if b < bsf_dp:
            cf_dp = 0
        elif bcol_dp > b >= bsf_dp:
            cf_dp = ((ccol_dp - csf_dp) / (bcol_dp - bsf_dp)) * (b - bsf_dp) + csf_dp
        else:
            cf_dp = ccol_dp

        if b < bsf_df:
            cf_df = 0
        elif bcol_df > b >= bsf_df:
            cf_df = ((ccol_df - csf_df) / (bcol_df - bsf_df)) * (b - bsf_df) + csf_df
        else:
            cf_df = ccol_df

        area_col = col_size**2
        moment_inertia_col = col_size**4/12

        k_col = Structure.stiffness(self, area=area_col, moment_inertia=moment_inertia_col, height=L)

        Mom, phi = Costs.compression(self, col_size=col_size, L=L)
        ddi = Costs.deformation_damage_index(self, B=b, stiffness=k_col, Mom=Mom, phi=phi)

        DDI1 = self.cost["DDI_1"]
        DDI2 = self.cost["DDI_2"]
        DDI3 = self.cost["DDI_3"]
        DDI4 = self.cost["DDI_4"]
        cDDI1 = self.cost["cost_DDI_1"]
        cDDI2 = self.cost["cost_DDI_2"]
        cDDI3 = self.cost["cost_DDI_3"]
        cDDI4 = self.cost["cost_DDI_4"]

        if ddi < DDI1:
            cf_duc = 0
        elif DDI1 <= ddi < DDI2:
            bsf = DDI1
            bcol = DDI2
            csf = cDDI1
            ccol = cDDI2
            cf_duc = ((ccol - csf) / (bcol - bsf)) * (ddi - bsf) + csf

        elif DDI2 <= ddi < DDI3:
            bsf = DDI2
            bcol = DDI3
            csf = cDDI2
            ccol = cDDI3
            cf_duc = ((ccol - csf) / (bcol - bsf)) * (ddi - bsf) + csf

        elif DDI3 <= ddi < DDI4:
            bsf = DDI3
            bcol = DDI4
            csf = cDDI3
            ccol = cDDI4
            cf_duc = ((ccol - csf) / (bcol - bsf)) * (ddi - bsf) + csf
        else:
            cf_duc = cDDI4

        f_duc = cf_duc * ncolumns

        cf = cf_cs + cf_duc + (cf_eg + cf_dp + cf_df)*0 #Only considering the structural damage

        return cf
