"""Kill owned model servers when the parent process exits, including forced exit."""
import os

class ProcessJob:
    def __init__(self,process):
        self.handle=None
        if os.name!='nt':return
        import ctypes as c
        from ctypes import wintypes as w
        class BASIC(c.Structure):
            _fields_=[('PerProcessUserTimeLimit',c.c_int64),('PerJobUserTimeLimit',c.c_int64),('LimitFlags',w.DWORD),('MinimumWorkingSetSize',c.c_size_t),('MaximumWorkingSetSize',c.c_size_t),('ActiveProcessLimit',w.DWORD),('Affinity',c.c_size_t),('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]
        class IO(c.Structure):_fields_=[(name,c.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class EXTENDED(c.Structure):
            _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',c.c_size_t),('JobMemoryLimit',c.c_size_t),('PeakProcessMemoryUsed',c.c_size_t),('PeakJobMemoryUsed',c.c_size_t)]
        self.api=c.WinDLL('kernel32',use_last_error=True)
        self.api.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];self.api.CreateJobObjectW.restype=w.HANDLE
        self.api.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
        self.api.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
        self.api.CloseHandle.argtypes=[w.HANDLE]
        handle=self.api.CreateJobObjectW(None,None)
        if not handle:raise c.WinError(c.get_last_error())
        info=EXTENDED();info.BasicLimitInformation.LimitFlags=0x2000
        if not self.api.SetInformationJobObject(handle,9,c.byref(info),c.sizeof(info)) or not self.api.AssignProcessToJobObject(handle,w.HANDLE(int(process._handle))):
            error=c.get_last_error();self.api.CloseHandle(handle);raise c.WinError(error)
        self.handle=handle
    def close(self):
        if self.handle:self.api.CloseHandle(self.handle);self.handle=None
