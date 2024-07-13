from environment import (Buffer, Clock, CostConfig, ExecutionTimeSampler,
                         GammaDistributionSampler, GaussianTimeSampler,
                         InputNode, MISOFusionTask, ProcessingNode,
                         SISOComputeTask, System, SystemBuilder, TimeProvider)


class ThreeStageSystemBuilder(SystemBuilder):
    def __init__(self, time_provider: TimeProvider):
        super().__init__(time_provider)
        self._local_idx = -1
        self._input_idx = 0

    def set_global_node(self, sampler: ExecutionTimeSampler, rejection_threshold: int = 100):
        compute_task = MISOFusionTask(self._time_provider, sampler, rejection_threshold)
        node = ProcessingNode(compute_task, "GLOBAL", Buffer(id="GLOBAL_OUT"))
        self.system.add_processing_node(node, None, is_state_node=True, is_action_node=True)

    def add_local_node(
        self,
        sampler: ExecutionTimeSampler,
        rejection_threshold: int = 100,
    ):
        self._local_idx += 1
        self._input_idx = 0
        compute_task = MISOFusionTask(self._time_provider, sampler, rejection_threshold)
        node = ProcessingNode(compute_task, f"LOCAL_{self._local_idx}", Buffer(id=f"LOCAL_{self._local_idx}_OUT"))
        self._system.add_processing_node(node, "GLOBAL", is_state_node=True, is_action_node=True)

    def add_input_module(
        self,
        input_proc_sampler: ExecutionTimeSampler,
        input_sens_sampler: ExecutionTimeSampler,
        t0_input=0,
        tau_input=100,
    ):
        if self._local_idx < 0:
            raise ValueError("Cannot add an input module, because no local node has been added yet.")

        # Create a worker that processes the input
        compute_task = SISOComputeTask(self._time_provider, input_proc_sampler)
        input_proc_node = ProcessingNode(
            compute_task,
            f"INPUT_PROC_{self._local_idx}{self._input_idx}",
            Buffer(id=f"INPUT_PROC_{self._local_idx}{self._input_idx}_OUT"),
        )
        self.system.add_processing_node(
            input_proc_node, f"LOCAL_{self._local_idx}", is_state_node=True, is_action_node=False
        )

        # Create an input that automatically triggers the worker
        input_sens_node = InputNode(
            self._time_provider,
            tau_input,
            t0_input,
            input_sens_sampler,
            f"INPUT_SENS_{self._local_idx}{self._input_idx}",
            Buffer(id=f"INPUT_SENS_{self._local_idx}{self._input_idx}_OUT"),
        )
        self.system.add_input_node(
            input_sens_node, f"INPUT_PROC_{self._local_idx}{self._input_idx}", trigger_next_compute=True
        )
        self._input_idx += 1


class ThreeStageSystemDirector(object):
    def __init__(self, time_provider: TimeProvider):
        self._time_provider = time_provider

    def build_simple_system(
        self,
        num_local=3,
        num_input_per_local=4,
        global_sampler=GammaDistributionSampler(7.5, 1.0, 4.0, 40.0),
        local_sampler=GammaDistributionSampler(9.0, 0.5, 2.0, 20.0),
        input_proc_sampler=GammaDistributionSampler(9.0, 0.5, 5.0, 64.0),
        input_sens_sampler=GaussianTimeSampler(5, 3, 0, 10),
        t0_input=0,
        tau_input=100,
    ) -> System:
        builder = ThreeStageSystemBuilder(self._time_provider)
        builder.set_global_node(global_sampler, tau_input),
        for i in range(num_local):
            builder.add_local_node(local_sampler, tau_input)
            for j in range(num_input_per_local):
                builder.add_input_module(
                    input_proc_sampler, input_sens_sampler, t0_input=t0_input, tau_input=tau_input
                )
        builder.system.compile()
        return builder.system
