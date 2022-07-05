#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from fieldmaptrack import FieldMap, Beam, Trajectory

class FieldmapOnAxisAnalysis:

    def __init__(self,
                 fieldmap=None,
                 filename=None,
                 beam_energy=3.0
                 ):
        """."""
        self.filename = filename
        self.fieldmap = fieldmap
        self.rz = None
        self.bx = None
        self.by = None
        self.bz = None
        self.beam = Beam(beam_energy)
        self.traj = None

    def calc_K(self, period, B, zmin, zmax, rz=None):
        begin_idx = np.where(rz==zmin)
        end_idx = np.where(rz==zmax)
        # length = end_idx[0][0] - begin_idx[0][0]
        k = 2*np.pi/(period)
        a11, a12, a21, a22 = [0, ]*4
        b1, b2 = 0, 0
        for i in np.arange(begin_idx[0][0], end_idx[0][0]+1, 1):
            b1 = B[i]*np.cos(k*rz[i]) + b1
            b2 = B[i]*np.sin(k*rz[i]) + b2
            a11 = (np.cos(k*rz[i]))**2 + a11
            a12 = np.cos(k*rz[i])*np.sin(k*rz[i]) + a12
            a22 = (np.sin(k*rz[i]))**2 + a22
        a21 = a12
        M = np.array([[a11,a12],[a21,a22]])
        det = np.linalg.det(M)
        M_inv = np.array([[a22,-a12],[-a21,a11]])/det
        B_matriz = np.array([[b1],[b2]])
        Res = np.dot(M_inv,B_matriz)
        a = Res[0][0]
        b = Res[1][0]
        B0 = np.sqrt(a**2+b**2)
        K = 93.36*B0*(period*1e-3)
        return K, B0

    def _calc_fx(self,
                 s_step=0.5, max_rz=4000, plot_flag=True,
                 ipos=[0,0,0], iang=[0,0,1], end=None):
        init_rx, init_ry, init_rz = ipos
        init_px, init_py, init_pz = iang
        self.traj.calc_trajectory(
            s_step=s_step, min_rz=max_rz, 
            init_rx=init_rx, init_ry=init_ry, init_rz=init_rz,
            init_px=init_px, init_py=init_py, init_pz=init_pz)
        z = self.traj.rz.copy()
        x = self.traj.rx.copy()
        px= self.traj.px.copy()
        dpx = np.diff(px)
        if end == None:
            idx_end = np.where(np.abs(dpx) < 1e-7)
            final_idx = []
            cond = int(max_rz/(2*s_step))
            for i in np.arange(np.shape(idx_end)[1]):
                if idx_end[0][i] > cond:
                    final_idx.append(idx_end[0][i])
            final = final_idx[0]
        else:
            final = end
        if plot_flag == True:
            plt.plot(z,x, color='g', label='step = {:.1f}mm'.format(s_step))
            plt.xlabel('Z coordinate (mm)')
            plt.ylabel('X coordinate (mm)')
            plt.legend()
            plt.show()
        return x[final], z[final], final, px[-1]

    def _calc_fy(self,
        s_step=0.5, max_rz=4000, plot_flag=True,
        ipos=[0,0,0], iang=[0,0,1], end=None):
        init_rx, init_ry, init_rz = ipos
        init_px, init_py, init_pz = iang
        self.traj.calc_trajectory(
            s_step=s_step, min_rz=max_rz,
            init_rx=init_rx, init_ry=init_ry, init_rz=init_rz,
            init_px=init_px, init_py=init_py, init_pz=init_pz)
        z = self.traj.rz.copy()
        y = self.traj.ry.copy()
        py= self.traj.py.copy()
        dpy = np.diff(py)
        if end == None:
            idx_end = np.where(np.abs(dpy) == 0)
            final_idx = []
            cond = int(max_rz/(2*s_step))
            for i in np.arange(np.shape(idx_end)[1]):
                if idx_end[0][i] > cond:
                    final_idx.append(idx_end[0][i])
            final = final_idx[0]
        else:
            final = end
        if plot_flag == True:
            plt.plot(z,y, color='g', label='step = {:.1f}mm'.format(s_step))
            plt.xlabel('Z coordinate (mm)')
            plt.ylabel('Y coordinate (mm)')
            plt.legend()
            plt.show()
        return y[final], z[final], final, py[-1]

    @staticmethod
    def calc_first_integral(B, rz=None):
        dz = 1e-3*(rz[-1] - rz[0]) / len(rz)
        integral = np.trapz(B, dx=dz)
        return integral

    @staticmethod
    def calc_second_integral(B, rz=None):
        dz = 1e-3*(rz[-1] - rz[0]) / len(rz)
        integral = []
        for i in np.arange(2, len(B), 1):
            integral.append(np.trapz(B[:i], dx=dz))
            array_integral = np.array(integral)
        integral2 = np.trapz(array_integral, dx=dz) 
        return integral2

    def load_fieldmap(self):
        field = FieldMap(self.fieldmap)
        beam = self.beam
        self.traj = Trajectory(beam=beam, fieldmap=field)
        # Get field components on axis
        idx_x0 = int((len(field.rx)-1)/2)
        idx_y0 = int((len(field.ry)-1)/2)
        self.z = field.rz
        self.Bx = field.bx[idx_x0,idx_y0, :]
        self.By = field.by[idx_x0,idx_y0, :]
        self.Bz = field.bz[idx_x0,idx_y0, :]


    def run(self):
        """."""
        x, z_end, idx_end, px = \
            self._calc_fx(s_step=0.5, max_rz=4000, plot_flag=False,
                ipos=[0,0,0], iang=[0,0,1])
        y, z_end, idx_end, py = \
            self._calc_fy(s_step=0.5, max_rz=4000, plot_flag=False,
                ipos=[0,0,0], iang=[0,0,1], end=idx_end)
    
        print("Final y position: {:.2f} um".format(1e3*y))
        print("Final y angle   : {:.2f} urad".format(1e6*py))
        print("Final x position: {:.2f} um".format(1e3*x))
        print("Final x angle   : {:.2f} urad".format(1e6*px))

        by_integral1 = self.calc_first_integral(B=self.By,z=self.z)
        print("By first integral (Tm)  : {:.3e}".format(by_integral1))

        by_integral2 = self.calc_second_integral(B=self.By,z=self.z)
        print("By second integral (Tm2): {:.3e}".format(by_integral2))

        bx_integral1 = self.calc_first_integral(B=self.Bx,z=self.z)
        print("Bx first integral (Tm)  : {:.3e}".format(bx_integral1))

        bx_integral2 = self.calc_second_integral(B=self.Bx,z=self.z)
        print("Bx second integral (Tm2): {:.3e}".format(bx_integral2))

        Ky, Bymax = self.calc_K(period=50,B=self.By,zmin=500,zmax=2500,z=self.z)
        print("K value for By          : ", Ky)
        print("Field amplitude for By  : ", Bymax)

        Kx, Bxmax = self.calc_K(period=50,B=self.Bx,zmin=500,zmax=2500,z=self.z)
        print("K value for Bx          : ", Kx)
        print("Field amplitude for Bx  : ", Bxmax)

        my_file = open(self.filename,"w") #w=writing
        my_file.write('x[um]\tpx[rad]\ty[mm]\tpy[urad]\tIx[Tm]\t        Iy[Tm]\t        I2x[Tm2]\tI2y[Tm2]\tBx[T]\tBy[T]\tKx\tKy\n')
        my_file.write("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t       {:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(1e3*x,1e6*px,1e3*y,1e6*py,bx_integral1,by_integral1,bx_integral2,by_integral2,Kx,Bxmax,Ky,Bymax))
        my_file.close()
        

