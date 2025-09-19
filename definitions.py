#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :
@File    : definitions.py
@Author  : igeng
@Date    : 2022/4/27 15:48 
@Descrip :
'''

class VM:
    def __init__(self, vm_id, cpu, mem):
        self.id = vm_id
        self.cpu = cpu
        self.mem = mem
        self.cpu_now = cpu
        self.mem_now = mem
        self.stop_use_clock = 0
        self.used_time = 0
        self.res_uti_cpu = 0
        self.res_uti_mem = 0

class JOB:
    def __init__(self, arrival_time, j_id, j_type, cpu, mem, ex, duration):
        self.arrival_time = arrival_time
        # self.start_time = None
        # self.finish_time = None

        self.start_time = 0
        self.finish_time = 0

        self.id = j_id
        self.type = j_type
        self.cpu = cpu
        self.mem = mem
        self.ex = ex
        self.ex_placed = 0
        self.ex_failed = 0
        self.duration = duration
        self.ex_placement_list = []
        self.running = False
        self.finished = False

        self.wait_time = 0

    def __lt__(self, other):
        return self.finish_time < other.finish_time