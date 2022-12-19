import math
import json
import argparse
import os
import numpy as np
from neuron import h, rxd
from neuron.units import hour, day, μm, ms, mV, mM
from neuron.rxd.node import Node3D
from matplotlib import pyplot as plt
import sqlite3
import pandas as pd

plt.ion()
h.load_file("stdrun.hoc")
rxd.options.ics_distance_threshold = -1e-15
rxd.options.concentration_nodes_3d = "all"


def boxoff(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


dendcount = 0


class Dendrite:
    """
    A 1D or hybrid 1D/3D model of a dendrite with 2N + 1 spines. The
    dendrite will be divided into three sections, with the central section
    and corresponding three spines simulated in 3D if dx is provided.

    Attributes
    ----------
    dend    list of 3 sections the left, center and right part of the dendrite.
    prot    rxd.Species representing the proteins synthesized by the
            active polyribosomes in the spine heads.

    """

    def __init__(
        self,
        Ls,
        N=3,
        dend_diam=5 * μm,
        dend_pad=0 * μm,
        neck_diam=0.2 * μm,
        neck_length=2 * μm,
        head_diam=1 * μm,
        head_length=1 * μm,
        D1=1e-3 * μm ** 2 / ms,
        D2=1e-3 * μm ** 2 / ms,
        lambd=60 * μm,
        ifactor=1.25,
        k=0.215e-4,
        cdis=2.0 * mM,
        kn=300,
        dx=None,
        load_init=None,
        nsegs=(25, 5),
    ):
        """
        Parameters
        ----------
        Ls          (float) the distance between spines (in μm).
        N           (int)   the number of spines on either side of the central
                            unpotentiated spine, so the total number of
                            spines is 2*N + 1.
        dend_diam   (float) the diameter (in μm) of the dendrite, default 5μm.
        dend_pad    (float) additional space on the ends of dendrite with no
                            spines. The total length will be;
                            (2 * N + 1) * Ls + 2 * dend_pad
        neck_diam   (float) the diameter (in μm) of the spine neck, default
                            1μm.
        neck_length (float) the length of the spine neck (in μm), default
                            2μm.
        head_diam   (float) the diameter of the spine head (in μm), default
                            1μm.
        head_length (float) the length of the spine head (in μm), default
                            1μm.
        D1          (float) the diffusion coefficient (μm**2/ms) of proteins
                            in the dendrite and unpotentiated spines.
                            default 1e-3.
        D2          (float) the diffusion coefficient (μm**2/ms) of proteins
                            in potentiated spines,
                            default 1e-3.
        lambd       (float) the length constant for the protein (in μm),
                            lambd=(D1/K)**0.5 where K is the protein degradation
                            rate, default 60μm
        k           (float) protein synthesis rate default 0.215e-4 mM/ms
        cdis        (float) the threshold concentration (mM) for Hill function
                            representing protein synthesis in the spine heads,
                            default 2mM.
        kn          (int)   Hill coefficient for the function representing
                            protein synthesis, default 300.
        dx          (float) optional voxel size for 3D simulation, if dx is
                            None the simulation uses 1D.
        load_init   (str)   optional, path to a json file with concentration for
                            each segment to use as their initial value. If
                            load_init is None, the potentiated spines heads
                            start with 2 * cdis, elsewhere at 0.85 * cdis
                            (an elevated value that is sufficiently small not
                            to effect the resulting steady-state).
        nsegs       (list)  pair of integers for the number of segments to use
                            for the spine necks and spine heads, default
                            (11,5).
        """

        global dendcount

        # split the dendrite into Left (1D) Center (1D or 3D) and Right (1D) sections
        dend = [
            h.Section(name="dendL%i" % dendcount),
            h.Section(name="dendC%i" % dendcount),
            h.Section(name="dendR%i" % dendcount),
        ]
        dendcount += 1

        if N < 1:
            raise Exception(
                "Dendrite must have at least 1 spine either side of the potentiated central spine"
            )

        # parameters
        self.dend = dend
        self.dend_pad = dend_pad
        self.length = math.ceil(Ls * (2 * N + 1) + 2 * dend_pad)
        self.dend_diam = dend_diam
        self.neck_l = neck_length
        self.neck_diam = neck_diam
        self.Ls = Ls
        self.k = k
        self.ifactor = ifactor
        self.kn = kn
        self.D1 = D1
        self.D2 = D2
        self.lambd = lambd
        self.head_diam = head_diam
        self.head_l = head_length
        self.tx = dx / 10.0 if dx else 0  # a small offset to ensure 3D sections
        # are joined together
        self.dx = dx
        self.cdis = cdis
        self.allsec = dend.copy()
        self.secs3d = [dend[1]]

        # create the dendrite
        dendx = 0
        dendy = np.round(-self.length / 2.0, 2)
        len3d = 3 * Ls
        lens = [(self.length - len3d) / 2.0, len3d, (self.length - len3d) / 2.0]
        for i, (sec, length) in enumerate(zip(dend, lens)):
            sec.pt3dclear()
            sec.nseg = max(1, int(2 * length + (2 * length / 3) % 3))
            sec.pt3dadd(dendx, dendy, 0, dend_diam)

            # add additional point for 3d/1d connection
            if sec not in self.secs3d and dx is not None:
                sec.pt3dadd(dendx, np.round(dendy + dx, 2), 0, dend_diam)
            dendy = np.round(dendy + length, 2)

            # add additional point for 3d/1d connection
            if sec not in self.secs3d and dx is not None:
                sec.pt3dadd(dendx, np.round(dendy - dx, 2), 0, dend_diam)
            sec.pt3dadd(dendx, dendy, 0, dend_diam)

        dend[1].connect(dend[0](1), 0)
        dend[2].connect(dend[1](1), 0)

        # add spines (only 3 will be 3D)
        self.Nspines = 0
        self.allsec = [dend[0], dend[1], dend[2]]
        self.secs3d = [dend[1]]
        self.allhead = []
        spines = []
        self.add_spine(ypos=0, nsegs=nsegs, use_3d=True)
        for i in range(1, N):
            self.add_spine(
                ypos=-float(i) * Ls, nsegs=nsegs, use_3d=(i == 1), spinelist=spines
            )
            self.add_spine(
                ypos=float(i) * Ls, nsegs=nsegs, use_3d=(i == 1), spinelist=spines
            )

        self.active_spines = spines
        # use 3D on the central 3 spines if dx was specified
        if dx:
            # 3D sim
            rxd.set_solve_type(self.secs3d, dimension=3)
            self.cyt = rxd.Region(self.allsec, nrn_region="i", dx=dx)
        else:
            self.cyt = rxd.Region(self.allsec, name="cyt", nrn_region="i")

        # load a previous solution is one was provided
        # otherwise start with all but the central spine in a potentiated state
        if load_init:
            self.load(load_init)
            init = lambda nd: self._loaded_vals[repr(nd.segment)]
        else:
            init = lambda nd: 2 * cdis if nd.segment.sec in self.allhead[1:] else 0

        # define the species
        pp = rxd.Species(self.cyt, d=D1, name="prot", charge=1, initial=init)

        head_vols = {}
        for sec in self.allsec:
            if sec in self.allhead:
                if sec in self.secs3d and dx is not None:
                    head_vols[sec] = 1.0 / sum(pp.nodes(sec).volume)
                else:
                    head_vols[sec] = 1.0 / sum(self.cyt.geometry.volumes1d(sec))
            else:
                head_vols[sec] = 0

        # parameter 1/vol for node in spine heads and zero elsewhere to control
        # the production rate -- attempt reduce difference between 1D and 3D
        # simulations due to volume differences from voxelization
        self.in_head = rxd.Parameter(
            self.cyt, name="vol", initial=lambda nd: 1 if nd.sec in self.allhead else 0
        )

        # degradation occurs everywhere
        K = D1 / lambd / lambd
        self.degrad = rxd.Rate(pp, -K * pp)

        # convert the rate from nA to mM/ms
        k_adj = k * ifactor * self.in_head
        self.prod = rxd.Rate(pp, k_adj * pp ** kn / (pp ** kn + cdis ** kn))
        self.prot = pp
        for sec in self.active_spines:
            for nd in pp.nodes(sec):
                nd.d = D2
        for sec in self.dend:
            for nd in pp.nodes(sec):
                nd.d = D1

    def dend_seg_by_ypos(self, ypos):
        """returns the segment in the dendrite for a given position (ypos)"""

        pos = self.length / 2.0 + ypos
        for sec in self.dend:
            if sec.L >= pos:
                return sec(pos / sec.L)
            pos -= sec.L

    def dend_ypos_by_seg(self, seg):
        """return the position in the dendrite for a given segment (seg)"""

        offset = 0
        for sec in self.dend:
            if seg in sec:
                return offset + seg.x * sec.L
            offset += sec.L

    def add_spine(self, ypos, nsegs, use_3d, spinelist=None):
        """add a spine to the dendrite at position (ypos)"""

        # create the sections
        head = h.Section(name="head%i" % self.Nspines)
        neck = h.Section(name="neck%i" % self.Nspines)
        self.Nspines += 1

        # 1D set geometry
        head.L = self.head_l
        head.diam = self.head_diam
        neck.L = self.neck_l
        neck.diam = self.neck_diam
        neck.nseg = nsegs[0]
        head.nseg = nsegs[1]

        # depth to ensure the spine is in the dendrite
        depth = (
            self.dend_diam - (self.dend_diam ** 2 - self.neck_diam ** 2) ** (0.5)
        ) / 2.0

        # 3D set geometry
        neckx0 = self.dend_diam / 2.0 - self.tx - depth
        neckx1 = neckx0 + self.neck_l
        headx0 = neckx1 - self.tx
        headx1 = neckx1 + self.head_l

        neck.pt3dclear()
        neck.pt3dadd(neckx0, ypos, 0, self.neck_diam)
        neck.pt3dadd(neckx1, ypos, 0, self.neck_diam)

        head.pt3dclear()
        head.pt3dadd(headx0, ypos, 0, self.neck_diam)
        head.pt3dadd(headx0, ypos, 0, self.head_diam)
        head.pt3dadd(headx1, ypos, 0, self.head_diam)

        # store the sections
        self.allsec.append(neck)
        self.allsec.append(head)
        self.allhead.append(head)
        if spinelist is not None:
            spinelist.append(neck)
            spinelist.append(head)

        if use_3d:
            self.secs3d += [neck, head]

        # connect them if using 1D
        if not self.dx or not use_3d:
            head.connect(neck(1))
            neck.connect(self.dend_seg_by_ypos(ypos), 0)

    def plot1d(self, fun=np.sum):
        """plot the concentration in the segments of the dendrite"""

        dend_x = [
            self.dend_ypos_by_seg(seg) - self.length / 2.0
            for sec in self.dend
            for seg in sec
        ]
        dend_pp = [seg.proti for sec in self.dend for seg in sec]
        plt.plot(dend_x, dend_pp, label="%1.2fms" % h.t)
        plt.xlabel("x (μm)")
        plt.ylabel("concentration (mM)")

    def implots(self, dr=None, vmin=None, vmax=None):
        """heat plot the 3D concentrations

        Arguments
        ---------
        dr (str)        optional, 'x', 'y', or 'z' to average over the given
                        direction, if not specified all three are shown in
                        subplots

        vmin (float)    optional, the minimum concentration of the plot
        vmax (float)    optional, the maximum concentration of the plot.
        """

        drlookup = {"x": 0, "y": 1, "z": 2}
        r = self.cyt
        data = np.nan * np.ones((max(r._xs) + 1, max(r._ys) + 1, max(r._zs) + 1))
        for nd in dendrite.prot.nodes:
            if isinstance(nd, Node3D):
                data[nd._i, nd._j, nd._k] = nd.value
        fig = plt.figure(dpi=200)
        if dr:
            plt.imshow(
                np.nanmean(data, drlookup[dr]), aspect="auto", vmin=vmin, vmax=vmax
            )
            plt.colorbar()
        else:
            for i in range(3):
                plt.subplot(3, 1, 1 + i)
                plt.imshow(np.nanmean(data, i), aspect="auto", vmin=vmin, vmax=vmax)
                plt.colorbar()

    def save(self, filename):
        """save the segment concentrations to a json file (filename) that can
        be used to initialize the model"""

        vals = dict()
        for sec in self.allsec:
            for seg in sec:
                vals[repr(seg)] = seg.proti
        with open(filename, "w") as fp:
            json.dump(vals, fp)

    def load(self, filename):
        """load the segment concentrations from a json file (filename)"""

        with open(filename, "r") as fp:
            self._loaded_vals = json.load(fp)

    def save_concentrations(self, filename):

        """save the protein concentration to filename"""
        np.save(filename, np.array(dendrite.prot.nodes.value))

    def save_sql(self, dbfilename):
        neck_nd = self.prot.nodes(dendrite.active_spines[0](1e-4))[0]
        dend_nd = self.prot.nodes(dendrite.active_spines[0](0))[0]

        neck = neck_nd.sec
        dend = dend_nd.sec
        neck_dx = neck.L / neck.nseg
        dend_dx = dend.L / dend.nseg
        area = np.pi * (neck.diam / 2) ** 2
        rate_out = 2 * neck_nd.d * area / (neck_nd.volume * neck_dx)
        rate_in = rate_out * neck_nd.volume / dend_nd.volume
        flux = rate_out * neck_nd.value - rate_in * dend_nd.value
        data = pd.DataFrame(
            {
                "Ls": self.Ls,
                "N": (self.Nspines + 1) / 2,
                "dend_diam": self.dend_diam,
                "dend_pad": self.dend_pad,
                "neck_diam": self.neck_diam,
                "head_diam": self.head_diam,
                "head_length": self.head_l,
                "neck_length": self.neck_l,
                "D1": [self.D1],
                "D2": [self.D2],
                "lambda": self.lambd,
                "k": self.k,
                "ifactor": self.ifactor,
                "cdis": self.cdis,
                "kn": self.kn,
                "dx": dx if self.dx is not None else 0,
                "active": dendrite.allhead[0].proti > self.cdis,
                "conc": dendrite.allhead[0].proti,
                "flux": flux,
            }
        )
        with sqlite3.connect(dbfilename) as conn:
            data.to_sql("data", conn, if_exists="append", index=False)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="""Run hybrid 1D/3D spines simulation"""
        )
        parser.add_argument(
            "--N",
            nargs="?",
            type=int,
            default=5,
            help="""number of spines either side of the central spine -- default 5""",
        )
        parser.add_argument(
            "--Ls",
            nargs="?",
            type=float,
            default=19,
            help="""length between spines -- default 19um""",
        )
        parser.add_argument(
            "--D2",
            nargs="?",
            type=float,
            default=1e-3,
            help="""Diffusion in potentiated spines -- default 1e-3um^2/ms""",
        )
        parser.add_argument(
            "--lambd",
            nargs="?",
            type=float,
            default=60,
            help="""length scale -- default 60um""",
        )
        parser.add_argument(
            "--k",
            nargs="?",
            type=float,
            default=0.215e-4,
            help="""protein production rate -- default 0.215e-4""",
        )
        parser.add_argument(
            "--dx",
            nargs="?",
            type=float,
            default=None,
            help="""3D discretization  -- default use 1D only""",
        )
        parser.add_argument(
            "--neck_length",
            nargs="?",
            type=float,
            default=2,
            help="""spine neck length  -- default 2um""",
        )
        parser.add_argument(
            "--initial",
            nargs="?",
            type=str,
            default=None,
            help="""json file for initial concentrations by segment""",
        )
        parser.add_argument(
            "--output",
            nargs="?",
            type=str,
            default=None,
            help="""output filename for summary data""",
        )
        args = parser.parse_args()
    except:
        os._exit(1)

    Ls = (
        args.Ls
    )  # critical value 19.475um (or 21.585um if the production rate is 25% greater)
    dx = args.dx
    N = args.N
    D2 = args.D2
    init_file = args.initial
    lambd = args.lambd
    rxd.nthread(4)

    cv = h.CVode()
    cv.active(True)
    cv.atol(1e-7)
    dendrite = Dendrite(
        N=N,
        Ls=Ls,
        dx=dx,
        D2=D2,
        k=args.k,
        neck_length=args.neck_length,
        lambd=lambd,
        load_init=init_file,
    )
    t_vec = h.Vector().record(h._ref_t)
    c_vec = h.Vector().record(dendrite.allhead[0](0.5)._ref_proti)

    # initialize and run
    h.finitialize(-70 * mV)
    dconc = dendrite.dend[1](0.5).proti
    hconc = dendrite.allhead[0](0.5).proti
    for i in range(1, 10):
        h.continuerun(i * day)
        if (
            abs(dconc - dendrite.dend[1](0.5).proti) < 1e-12
            and abs(hconc - dendrite.allhead[0](0.5).proti) < 1e-12
        ):
            break
        else:
            dconc = dendrite.dend[1](0.5).proti
            hconc = dendrite.allhead[0](0.5).proti
    else:
        print("did not converge")

    if args.output:
        dendrite.save_sql(args.output)

    # save the results
    if dx:
        dendrite.save_concentrations(
            "spines_hybrid_N_%i_Ls_%1.2f_dx_%1.2.npy" % (N, Ls, dx)
        )
        dendrite.plot1d()
        plt.savefig("spines_hybrid_N_%i_Ls_%1.2f_dx_%1.2_1D_plot.png" % (N, Ls, dx))
        plt.close()

        dendrite.implot()
        plt.savefig("spines_hybrid_N_%i_Ls_%1.2f_dx_%1.2_heatmap.png" % (N, Ls, dx))

    else:
        dendrite.save_concentrations("spines_1D_N_%i_Ls_%1.2f.npy" % (N, Ls))
        dendrite.plot1d()
        plt.savefig("spines_1D_N_%i_Ls_%1.2f_1D_plot.png" % (N, Ls))
        plt.close()
