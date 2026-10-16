"""Paper 1 native CarSim: common sprung-CG point and world-axis x/v/a.

Adapted from the established external backend. Only motion-sensor outputs are
added; no friction override or vehicle physics parameter changes are introduced.
"""
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

INSTALL=Path(r'E:\Applications\CarSim')
DLL=INSTALL/'Programs/solvers/carsim_64.dll'
ARCHIVE=INSTALL/'Resources/CPAR_Archives/SDK_External_Control.cpar'
MEMBER='Results/Run_5f3d2edd-31c1-4bd2-8d7d-422ce1ab2a71/Run_all.par'
FIELDS=('time','x','y','z','vx','vy','ax','ay','yaw','pitch','roll','xo','yo','zo','body_vx','body_ax')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out,speed,duration):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:raw=z.read(MEMBER)
    text=raw.decode('utf-8').replace('\r','').rstrip()
    assert text.endswith('END')
    commands=['RESET_IMPORTS','RESET_EXPORTS','DEFINE_UNITS bridge_mps2 1',f'TSTOP {duration:.12g}',
              'IPRINT 20','TSTEP 0.0005',f'SPEED_TARGET_CONSTANT {speed*3.6:.12g}',
              'OPT_SC 3','OPT_STOP 0','IMPORT IMP_THROTTLE_ENGINE REPLACE 0',
              'IMPORT IMP_PCON_BK REPLACE 0','IMPORT IMP_STEER_SW REPLACE 0',
              'ISENSOR_M = 1','DEFINE_MOTION_SENSORS 1 + ISENSOR_M - 1 - NSENSOR_M',
              'OPT_S_BODY_TYPE 1','OPT_S_BODY_ID 1','OPT_S_OUTPUT 4',
              'X_S -LX_CG_SU','Y_S Y_CG_SU','Z_S H_CG_SU',
              'A_ROLL_S 0','A_PITCH_S 0','A_YAW_S 0','MAKE_MOTION_SENSOR_OUTPUTS']
    expressions=[('Bt','T','s'),('Bx','X_S1','m'),('By','Y_S1','m'),('Bz','Z_S1','m'),
                 ('Bvx','VxN_S1','m/s'),('Bvy','VyN_S1','m/s'),('Bax','AxN_S1','bridge_mps2'),
                 ('Bay','AyN_S1','bridge_mps2'),('Byaw','Yaw','rad'),('Bpitch','Pitch','rad'),('Broll','Roll','rad'),
                 ('Bxo','Xo','m'),('Byo','Yo','m'),('Bzo','Zo','m'),('Bbodyv','Vx','m/s'),('Bbodya','Ax','bridge_mps2')]
    for name,expr,unit in expressions:
        commands.extend([f'DEFINE_OUTPUT {name} = {expr}; {unit}; Bridge {name}',f'EXPORT {name}'])
    (out/'vehicle.par').write_text(text[:-3]+'\n'+'\n'.join(commands)+'\nEND\n',encoding='utf-8')
    lines=['SIMFILE','PRODUCT_NAME CarSim','PROGDIR '+str(INSTALL),'DLLFILE '+str(DLL),
           'DATADIR '+str(out),'INPUT '+str(out/'vehicle.par'),'FILEBASE '+str(out/'vehicle'),
           'ECHO '+str(out/'vehicle_echo.par'),'FINAL '+str(out/'vehicle_end.par'),
           'LOGFILE '+str(out/'vehicle_log.txt'),'ERDFILE '+str(out/'vehicle.erg'),'END']
    (out/'vehicle.sim').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={'source_archive':str(ARCHIVE),'source_member':MEMBER,'source_member_sha256':hashlib.sha256(raw).hexdigest(),
              'sim_sha256':sha(out/'vehicle.sim'),'par_sha256':sha(out/'vehicle.par'),
              'solver_sha256':sha(DLL),'initial_speed_mps':speed,'duration_s':duration,
              'imports':['throttle_engine_fraction','brake_master_cylinder_MPa','steering_wheel_deg'],
              'exports_si':list(FIELDS),'units_status':'Verify native imports in resulting echo file; SI output formulas per VS Commands pp.44-45',
              'historical_results_reused':False,
              'state_reference':'Sprung CG motion sensor 1; inertial world-axis position, velocity and acceleration',
              'road_friction_override':False}
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return out


class Vehicle:
    def __init__(self,out,speed,duration):
        self.out=prepare(out,speed,duration)
        self.oldcwd=Path.cwd();os.chdir(self.out)
        self.api=C.CDLL(str(DLL));self.closed=False;self.initialized=False
        rp,ip=C.POINTER(C.c_double),C.POINTER(C.c_int)
        sigs={'vs_read_configuration':(None,[C.c_char_p,ip,ip,rp,rp,rp]),
              'vs_integrate_io':(C.c_int,[C.c_double,rp,rp]),'vs_copy_export_vars':(None,[rp]),
              'vs_error_occurred':(C.c_int,[]),'vs_get_error_message':(C.c_char_p,[]),
              'vs_terminate_run':(None,[C.c_double]),'vs_free_all':(None,[]),
              'vs_set_opt_error_dialog':(None,[C.c_int]),'vs_get_tstep':(C.c_double,[])}
        for name,(restype,argtypes) in sigs.items():
            f=getattr(self.api,name);f.restype=restype;f.argtypes=argtypes
        self.api.vs_set_opt_error_dialog(0)
        self.k=0;self.last_t=0.;self.last_command=(0.,0.,0.)
        try:
            ni,ne=C.c_int(),C.c_int();start,stop,dt=C.c_double(),C.c_double(),C.c_double()
            self.api.vs_read_configuration(str(self.out/'vehicle.sim').encode('mbcs'),C.byref(ni),C.byref(ne),C.byref(start),C.byref(stop),C.byref(dt))
            self.check();self.initialized=True
            self.start,self.stop,self.dt=start.value,stop.value,dt.value
            if ni.value!=3 or ne.value!=len(FIELDS):raise RuntimeError(f'I/O counts {ni.value}/{ne.value} differ')
            if not math.isclose(self.dt,.0005,abs_tol=1e-10):raise RuntimeError('Unexpected native T_DT')
            self.imports=(C.c_double*3)();self.exports=(C.c_double*len(FIELDS))()
            self.api.vs_copy_export_vars(self.exports);self.state=dict(zip(FIELDS,self.exports))
            if abs(self.state['vx']-speed)>.01:raise RuntimeError(f'Initial speed {self.state["vx"]} differs from {speed}')
            self.metadata={'dt_s':self.dt,'initial_state':self.state.copy(),'solver_dll':str(DLL),'solver_sha256':sha(DLL)}
            for name in ['vs_get_version_model','vs_get_version_product']:
                f=getattr(self.api,name);f.restype=C.c_char_p;f.argtypes=[];self.metadata[name]=f().decode(errors='replace')
        except Exception:
            self.close();raise

    def check(self):
        if self.api.vs_error_occurred():
            msg=self.api.vs_get_error_message()
            raise RuntimeError(msg.decode('utf-8',errors='replace') if msg else 'Unspecified VS Solver error')

    def sample(self,command):
        if self.closed:raise RuntimeError('Solver closed')
        throttle,brake,steer=map(float,command)
        if not all(math.isfinite(x) for x in command) or not 0<=throttle<=1 or not 0<=brake<=15 or abs(steer)>500:
            raise ValueError('Native actuator input outside declared test limits')
        t=self.start+self.k*self.dt
        if t>self.stop+1e-9:raise RuntimeError('Native solver horizon exceeded')
        for i,val in enumerate(command):self.imports[i]=val
        previous=self.last_command
        rc=self.api.vs_integrate_io(t,self.imports,self.exports);self.check()
        values=list(self.exports)
        if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite solver state')
        self.state=dict(zip(FIELDS,values))
        if abs(self.state['time']-t)>1e-8:raise RuntimeError('Native exported clock mismatch')
        self.last_t=t;self.k+=1;self.last_command=tuple(command)
        return dict(self.state,call_time=t,throttle_fraction=throttle,brake_mpa=brake,steer_deg=steer,
                    previous_throttle_fraction=previous[0],previous_brake_mpa=previous[1],stop_requested=bool(rc))

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.initialized:self.api.vs_terminate_run(self.last_t)
            else:self.api.vs_free_all()
        finally:os.chdir(self.oldcwd)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


POINTERS={
 'gear_status':('SV_GEAR_STAT',1.),'gear_command':('SV_GEAR',1.),
 'gear_delayed':('SV_GEAR_DELAY',1.),'gear_ratio':('SV_R_GEAR_SHIFT',1.),
 'engine_rpm':('SV_AV_ENG',30/math.pi),
 'brake_front_actual_mpa':('SV_PBK_L1',1e-6),
 'brake_rear_actual_mpa':('SV_PBK_L2',1e-6),
 'delayed_throttle_state':('SV_THROTTLE',1.),
 'shift_start_time':('SV_T_SHIFT_START',1.),
 'shift_active':('SV_TH_AUTO_SHIFT',1.)}

class TelemetryVehicle(Vehicle):
    def __init__(self,*args):
        super().__init__(*args)
        f=self.api.vs_get_var_ptr2;f.restype=C.POINTER(C.c_double)
        f.argtypes=[C.c_char_p,C.POINTER(C.c_int),C.c_int]
        self.pointers={}
        for key,(symbol,scale) in POINTERS.items():
            ptr=f(symbol.encode(),None,0)
            if not ptr:self.close();raise RuntimeError('Missing diagnostic: '+symbol)
            self.pointers[key]=(ptr,scale)
        self.metadata['diagnostic_channels']=POINTERS
        self.state.update(self.telemetry())
    def telemetry(self):return {k:float(p[0])*scale for k,(p,scale) in self.pointers.items()}
    def sample(self,command):
        row=super().sample(command);data=self.telemetry()
        row.update(data);self.state.update(data);return row
