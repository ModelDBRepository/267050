import math
import json
import argparse
import os
import numpy as np
from neuron import h, rxd
from neuron.units import hour, day, s, μm, ms, mV, mM
from neuron.rxd.node import Node3D
from matplotlib import pyplot as plt

plt.ion()
h.load_file("stdrun.hoc")


def boxoff(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


dendcount = 0


class Dendrite:
    """
    A 1D a dendrite with clusters of spines.

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
        n=2,
        C=3,
        Lc=20 * μm,
        dend_diam=5 * μm,
        dend_pad=0 * μm,
        neck_diam=0.2 * μm,
        neck_length=2 * μm,
        head_diam=1 * μm,
        head_length=1 * μm,
        D=1e-3 * μm ** 2 / ms,
        lambd=120 * μm,
        k=0.215e-4,
        ifactor=1.25,
        cdis=2.0 * mM,
        kn=300,
        load_init=None,
        nsegs=(11, 5),
    ):
        """
        Parameters
        ----------
        Ls          (float) the distance between spines (in μm).
        N           (int)   total number of spines
        n           (int)   number of potentiated clusters
        C           (int)   number of clusters
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
        D           (float) the diffusion coefficient (μm**2/ms) of proteins
                            synthesized by polyribosomes in the spine heads,
                            default 1e-3.
        lambd       (float) the length constant for the protein (in μm),
                            lambd=(D/K)**0.5 where K is the protein degradation
                            rate, default 60μm
        k           (float) protein synthesis rate (mM/ms) default
                            0.215e-4 mM/ms/μm**3
        cdis        (float) the threshold concentration (mM) for Hill function
                            representing protein synthesis in the spine heads,
                            default 2mM.
        kn          (int)   Hill coefficient for the function representing
                            protein synthesis, default 300.
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
        dend = [h.Section(name=f"dend{dendcount}_{i}") for i in range(C)]
        dendcount += 1

        if N < 1:
            raise Exception(
                "Dendrite must have at least 1 spine either side of the potentiated central spine"
            )

        # parameters
        self.dend = dend
        self.length = math.ceil(C * (Ls * N + Lc) + 2 * dend_pad)

        self.dend_diam = dend_diam
        self.neck_l = neck_length
        self.neck_diam = neck_diam
        self.Ls = Ls
        self.Lc = Lc
        self.C = C
        self.head_diam = head_diam
        self.head_l = head_length
        # are joined together
        self.allsec = dend.copy()
        self.allhead = []
        self.Nspines = 0
        self.nactive = n
        self.cdis = cdis
        self.active = []
        # create the dendrite
        dendx = 0
        dendy = np.round(-self.length / 2.0, 2)
        for i, sec in enumerate(dend):

            sec.pt3dclear()
            sec.nseg = min(
                max(1, int(2 * self.length + (2 * self.length / 3) % 3)), 32767
            )
            sec.pt3dadd(dendx, dendy, 0, dend_diam)
            dendy += N * Ls + Lc + (i == 0 or i == C - 1) * dend_pad
            sec.pt3dadd(dendx, dendy, 0, dend_diam)
            spiney = dendy - (N * Ls + Lc / 2 - Ls / 2 + (i == C - 1) * dend_pad)
            active = (C - n) / 2 <= i < (C + n) / 2
            for j in range(N):
                head = self.add_spine(ypos=spiney, nsegs=nsegs)
                if active:
                    self.active.append(head)
                spiney += Ls

        # connect the dendrite sections
        for i in range(1, C):
            dend[i].connect(dend[i - 1](1), 0)

        self.cyt = rxd.Region(self.allsec, nrn_region="i")

        # load a previous solution is one was provided
        # otherwise start with all but the central spine in a potentiated state
        if load_init:
            self.load(load_init)
            init = lambda nd: self._loaded_vals[repr(nd.segment)]
        else:
            # n spine on the end is unpotentiated
            init = lambda nd: 2.0 * cdis if nd.segment.sec in self.active else 0

        # define the species
        pp = rxd.Species(self.cyt, d=D, name="prot", charge=1, initial=init)

        head_vols = {}
        for sec in self.allsec:
            if sec in self.allhead:
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
        K = D / lambd / lambd
        self.degrad = rxd.Rate(pp, -K * pp)

        # convert the rate from nA to mM/ms
        k_adj = k * ifactor * self.in_head
        self.prod = rxd.Rate(pp, k_adj * pp ** kn / (pp ** kn + cdis ** kn))
        self.prot = pp
        self.headvec = h.Vector()
        self.headvec.record(pp.nodes(self.allhead[N - n - 1](0.5))._ref_value, 5 * s)
        self.head0vec = h.Vector()
        self.head0vec.record(pp.nodes(self.allhead[0](0.5))._ref_value, 5 * s)

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

    def add_spine(self, ypos, nsegs, use_3d=False):
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
        depth = 0

        # 3D set geometry
        neckx0 = self.dend_diam / 2.0 - depth
        neckx1 = neckx0 + self.neck_l
        headx0 = neckx1
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

        # connect them
        head.connect(neck(1))
        neck.connect(self.dend_seg_by_ypos(ypos), 0)
        return head

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

    def save_summary(self, dbfilename):
        import sqlite3
        import pandas as pd

        data = pd.DataFrame(
            {
                "nactive": self.nactive,
                "C": self.C,
                "N": self.Nspines / self.C,
                "L": self.Ls,
                "Lc": self.Lc,
                "dend_diam": self.dend_diam,
                "dend_l": sum([sec.L for sec in self.dend]),
                "neck_l": self.neck_l,
                "head_diam": self.head_diam,
                "head_l": self.head_l,
                "concentration": self.headvec[-1],
                "upstate": self.headvec[-1] > self.cdis,
            },
            index=[f"{self.nactive}_{self.Nspines}_{self.Ls}"],
        )
        with sqlite3.connect(dbfilename) as conn:
            data.to_sql("summary", conn, if_exists="append", index=False)
        print("save", data)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="""Run 1D simulations with clusters of spines."""
        )
        parser.add_argument(
            "--N",
            nargs="?",
            type=int,
            default=5,
            help="""number of spines in a cluster -- default 5""",
        )
        parser.add_argument(
            "--nactive",
            nargs="?",
            type=int,
            default=1,
            help="""number of potentiated clusters -- default 1""",
        )

        parser.add_argument(
            "--C",
            nargs="?",
            type=int,
            default=3,
            help="""number of clusters -- default 5""",
        )

        parser.add_argument(
            "--Ls",
            nargs="?",
            type=float,
            default=5,
            help="""length between spines -- default 5um""",
        )

        parser.add_argument(
            "--Lc",
            nargs="?",
            type=float,
            default=15,
            help="""length between clusters -- default 15um""",
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
            "--initial",
            nargs="?",
            type=str,
            default=None,
            help="""json file for initial concentrations by segment""",
        )
        parser.add_argument(
            "--summary",
            nargs="?",
            type=str,
            default=None,
            help="""a file to store if the spine was potentiated, if provided
                    no other output will be given""",
        )

        args = parser.parse_args()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)

    Ls = args.Ls
    Lc = args.Lc
    N = args.N
    init_file = args.initial
    n = args.nactive
    summary = args.summary
    rxd.nthread(4)

    cv = h.CVode()
    cv.active(True)
    cv.atol(1e-6)
    dendrite = Dendrite(
        N=N,
        n=n,
        Ls=Ls,
        load_init=init_file,
        C=args.C,
        Lc=Lc,
        lambd=args.lambd,
        k=args.k,
        dend_pad=100,
    )
    # initialize and run
    h.finitialize(-70 * mV)
    for i in range(1, 50):
        h.continuerun(i * day)
        if abs(dendrite.head0vec[-1] - dendrite.head0vec[-2]) / (5 * s) < 1e-9:
            break
    else:
        print(f"did not reach steady state in {h.t/day} days.")

    if summary is not None:
        dendrite.save_summary(summary)
    else:
        # save the results
        dendrite.save_concentrations(
            "cluster_1D_N_%i_Ls_%1.2f_Lc_%1.2f.npy" % (N, Ls, Lc)
        )
        dendrite.plot1d()
        plt.savefig("cluster_1D_N_%i_Ls_%1.2f_Lc_%1.2f_plot.png" % (N, Ls, Lc))
        plt.close()
