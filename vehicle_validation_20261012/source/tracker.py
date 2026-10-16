"""Gear-aware empirical inverse, anti-windup PI and explicit reference governor.
No theorem assurance; raw/gov/realized accelerations are distinct logged channels.
"""
import json
from pathlib import Path
import numpy as np

MODEL=Path(__file__).resolve().parents[1]/'inputs/actuator_model_v2.json'

class DynamicTracker:
 def __init__(self,kp=.5,ki=1.2,governed=True):
  self.model=json.loads(MODEL.read_text());self.kp=kp;self.ki=ki;self.governed=governed
  self.integral=0.;self.reference=None;self.last_mode=None
 def predicted(self,gear,v,throttle,p=0.):
  key=str(int(round(gear)));supported=key in self.model['models_by_gear']
  if not supported:key=min(self.model['models_by_gear'],key=lambda k:abs(int(k)-gear))
  support=self.model['models_by_gear'][key]['speed_support_mps']
  supported=supported and support[0]<=v<=support[1]
  c=self.model['models_by_gear'][key]['coefficients'];v=v/20
  return sum(x*y for x,y in zip(c,[1,v,v*v,throttle,throttle*throttle,throttle*v,p])),key,supported
 def invert(self,desired,s):
  coast,key,supported=self.predicted(s['gear_command'],s['vx'],0.)
  c=self.model['models_by_gear'][key]['coefficients']
  if desired<coast:
   p=(desired-coast)/c[-1];return (0.,float(np.clip(p,0,3)),0.),p>3,'brake',supported
  us=np.linspace(0,.5,101)
  axes=np.maximum.accumulate([self.predicted(s['gear_command'],s['vx'],float(u))[0] for u in us])
  throttle=float(np.interp(desired,axes,us))
  return (throttle,0.,0.),bool(desired>axes[-1]),'throttle',supported
 def command(self,raw,s,dt=.01):
  if self.reference is None:self.reference=s['ax']
  old=self.reference
  if self.governed:
   target=float(np.clip(raw,-3,1.5))
   target=max(target,-max(s['vx'],0)/.6)
   self.reference=old+float(np.clip(target-old,-2*dt,2*dt))
  else:self.reference=raw
  rate=(self.reference-old)/dt
  error=self.reference-s['ax'];integral=float(np.clip(self.integral+error*dt,-2,2))
  shift=abs(s['gear_command']-s['gear_delayed'])>1e-5 or s['time']-s['shift_start_time']<.3
  desired=self.reference+self.model['reference_compensation_tau_s']*rate+self.kp*error+self.ki*integral
  command,saturated,mode,supported=self.invert(desired,s)
  # No hidden resets of plant state. Freeze integration during gear transient
  # and when error pushes further beyond the empirical actuation range.
  if not shift and not saturated:self.integral=integral
  stopped=s['vx']<.2 and raw<0
  if stopped:command=(0.,1.5,0.)
  domain=3<=s['vx']<=22 and not shift and supported and not stopped and s['time']>=.3
  return command,{'raw_request':raw,'governed_reference':self.reference,'reference_rate':rate,
    'reference_change':self.reference-raw,'inverse_target':desired,'tracker_integral':self.integral,
    'saturated':saturated,'gear_transient':shift,'empirical_domain_valid':domain,
    'stopping_hold':stopped,'certificate_available':False}
