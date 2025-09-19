#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :
@File    : cluster.py
@Author  : igeng
@Date    : 2022/4/27 15:47 
@Descrip :
'''
import numpy as np
import definitions as defs
import workload
import copy

# cluster resource details

# 3种类型虚拟机，
vm_types = 3
vm1_total = 5
vm1_cpu = 3200
vm1_mem = 3200
vm2_total = 5
vm2_cpu = 3200
vm2_mem = 6400
vm3_total = 5
vm3_cpu = 6400
vm3_mem = 12800


job_features = 4
features = (vm1_total + vm2_total + vm3_total) * 2 + job_features
JOBS = []
VMS = []

cluster_state_init = []

max_episode_cost = 0
min_avg_job_duration = 0

def gen_cluster_state_lb(job_idx, jobs, vms):
    cluster_state = []
    i = 0
    cpu_total = 0
    mem_total = 0
    res_uti_cpu_diff2 = 0
    res_uti_mem_diff2 = 0
    while i < len(vms):
        cluster_state.append(1 - float(vms[i].cpu_now / vms[i].cpu))
        cluster_state.append(1 - float(vms[i].mem_now / vms[i].mem))
        cpu_total += 1 - float(vms[i].cpu_now / vms[i].cpu)
        mem_total += 1 - float(vms[i].mem_now / vms[i].mem)
        i += 1
    # cluster_state.append(jobs[job_idx].id)
    cpu_load_mu = cpu_total / len(vms)
    mem_load_mu = mem_total / len(vms)
    for i in range(len(vms)):
        res_uti_cpu_diff2 += np.abs((1 - float(vms[i].cpu_now / vms[i].cpu) - cpu_load_mu))
        res_uti_mem_diff2 += np.abs((1 - float(vms[i].mem_now / vms[i].mem) - mem_load_mu))
    # 集群中所有虚拟机负载的标准差
    cpu_sigma_pow2 = res_uti_cpu_diff2 / len(vms)
    mem_sigma_pow2 = res_uti_mem_diff2 / len(vms)

    cluster_state.append(jobs[job_idx].type)
    cluster_state.append(jobs[job_idx].cpu)
    cluster_state.append(jobs[job_idx].mem)
    cluster_state.append(jobs[job_idx].ex - jobs[job_idx].ex_placed - jobs[job_idx].ex_failed)


    return cluster_state

def gen_cluster_state(job_idx, jobs, vms):
    cluster_state = []
    i = 0
    while i < len(vms):
        cluster_state.append(vms[i].cpu_now)
        cluster_state.append(vms[i].mem_now)
        i += 1
    # cluster_state.append(jobs[job_idx].id)
    cluster_state.append(jobs[job_idx].type)
    cluster_state.append(jobs[job_idx].cpu)
    cluster_state.append(jobs[job_idx].mem)
    cluster_state.append(jobs[job_idx].ex - jobs[job_idx].ex_placed - jobs[job_idx].ex_failed)
    return cluster_state

def init_jobs():
    global JOBS
    JOBS = copy.deepcopy(workload.JOBS_WORKLOAD)

def init_vms():
    global VMS
    VMS = []
    for i in range(vm1_total):
        VMS.append(defs.VM(len(VMS), vm1_cpu, vm1_mem))
    for j in range(vm2_total):
        VMS.append(defs.VM(len(VMS), vm2_cpu, vm2_mem))
    for k in range(vm3_total):
        VMS.append(defs.VM(len(VMS), vm3_cpu, vm3_mem))

def init_cluster():
    init_jobs()
    init_vms()
    global cluster_state_init

    cluster_state_init = gen_cluster_state_lb(0, JOBS, VMS)


init_cluster()
