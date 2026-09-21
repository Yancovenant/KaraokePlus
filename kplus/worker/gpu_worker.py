import os
from pathlib import Path

from kplus import env


class GPUWorker:
    def __init__(self):
        pass


class KaggleWorker(GPUWorker):
    def __init__(self, api_token: str):
        env.kaggle  # noqa: B018
        os.environ["KAGGLE_API_TOKEN"] = api_token
        from kaggle import api
        self.api = api
        super().__init__()

    def get_quota(self) -> dict:
        quotas = self.api.quota_view()
        quotas_per_type = [
            {
                "gpu": {
                    "time_used": quotas.gpu_quota.time_used,
                    "time_allowed": quotas.gpu_quota.total_time_allowed
                }
            },
            {
                "tpu": {
                    "time_used": quotas.tpu_quota.time_used,
                    "time_allowed": quotas.tpu_quota.total_time_allowed
                }
            }
        ]
        return {
            "refresh_time": quotas.quota_refresh_time,
            "types": quotas_per_type
        }

    def run(self, folder: str):
        """ Run kaggle kernel instance and push the notebook.
        
            :param str folder: the folder path containing
                               both .ipynb and kernel-metadata.json.
        """
        if not folder:
            raise RuntimeError("folder containing metadata and notebook need to be specified")
        result = self.api.kernels_push(
            folder=folder,
            timeout=None,
            acc=""
        )
        if result is None:
            raise RuntimeError("Kaggle push error, no definite data, see previous output.")
        elif not result.error:
            if result.invalidTags:
                raise RuntimeError(
                    "The following are not valid tags and could not be added "
                    "to the kernel: " + str(result.invalidTags)
                )
            if result.invalidDatasetSources:
                raise RuntimeError(
                    "The following are not valid dataset sources and could not "
                    "be added to the kernel: " + str(result.invalidDatasetSources)
                )
            if result.invalidCompetitionSources:
                raise RuntimeError(
                    "The following are not valid competition sources and could "
                    "not be added to the kernel: " + str(result.invalidCompetitionSources)
                )
            if result.invalidKernelSources:
                raise RuntimeError(
                    "The following are not valid kernel sources and could not "
                    "be added to the kernel: " + str(result.invalidKernelSources)
                )
            return result
        else:
            raise RuntimeError(f"Kaggle push error: {result.error}")


if __name__ == "__main__":
    from kplus.tools import rich
    print("=== Kaggle Worker Test ===")
    worker = KaggleWorker(api_token="KGAT_f0254f4f2f54aee972cbc2b638a4ff89") # this already exposed, need to rotate
    # rich.inspect(worker.api, methods=True)
    # print(worker.api._authenticate_with_access_token())
    print(worker.api.print_config_values())

    print("1. Quota View")
    quotas = worker.api.quota_view()
    for quota_type in [quotas.gpu_quota, quotas.tpu_quota]:
        print(
            f"Type of `{type(quota_type)}` Quota\n"
            f"has_ever_run              : {quota_type.has_ever_run} - {type(quota_type.has_ever_run)}\n"
            f"is_pay_to_scale_enabled   : {quota_type.is_pay_to_scale_enabled} - {type(quota_type.is_pay_to_scale_enabled)}\n"
            f"minimum_time_allowed      : {quota_type.minimum_time_allowed} - {type(quota_type.minimum_time_allowed)}\n"
            f"time_reserved             : {quota_type.time_reserved} - {type(quota_type.time_reserved)}\n"
            f"time_used:                : {quota_type.time_used} - {type(quota_type.time_used)}\n"
            f"total_time_allowed        : {quota_type.total_time_allowed} - {type(quota_type.total_time_allowed)}\n"
        )
    print(
        "`quotas.quota_refresh_time` \n"
        f"quota_refresh_time : {quotas.quota_refresh_time}"
    )

    print("2. Kernels")
    kernels = worker.api.kernels_list(mine=True)
    rich.inspect(kernels)

    print("4. Kernel Output")
    outputs = worker.api.kernels_output(
        "burninfist/kplus-gpu-worker",
        path=None
    )
    rich.print(outputs)

    print("3. Kernels Push")
    response = worker.run()
    rich.print(response)
