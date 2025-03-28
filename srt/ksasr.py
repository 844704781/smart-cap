import json

import requests
from typing import List

from .ASRDataSeg import ASRDataSeg
from .BaseASR import BaseASR
import logging
logger = logging.getLogger(__name__)

class KuaiShouASR(BaseASR):
    def __init__(self, audio_path: [str, bytes], use_cache: bool = False):
        super().__init__(audio_path, use_cache)

    def _run(self) -> dict:
        return self._submit()

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        return [ASRDataSeg(u['text'], u['start_time'], u['end_time']) for u in resp_data['data']['text']]

    def _submit(self) -> dict:
        payload = {
            "typeId": "1"
        }
        files = [('file', ('test.mp3', self.file_binary, 'audio/mpeg'))]
        logger.info(f"Submitting audio file to KuaiShou ASR...")
        result = requests.post("https://ai.kuaishou.com/api/effects/subtitle_generate", data=payload, files=files,
                               timeout=(30, 120))
        logger.info(f"KuaiShou ASR submitted: {result.text},status code: {result.status_code}")
        if not result.ok:
            raise Exception(f"Error: {result.status_code} {result.text}")

        resp = result.json()
        if resp['code'] != 200:
            raise Exception(f"Error: {resp['code']} {resp['msg']}")
        for item in resp['data'].get('text'):
            if not item.get('end_time'):
                item['end_time'] = 0
            if not item.get('start_time'):
                item['start_time'] = 0
            if not item.get('text'):
                item['text'] = ''
            item['end_time'] = item['end_time'] * 1000
            item['start_time'] = item['start_time'] * 1000
        return resp


if __name__ == '__main__':
    audio_file = r"extracted_audio.wav"
    asr = KuaiShouASR(audio_file)
    asr_data = asr.run()

    srt = asr_data.to_srt()
    print(srt)
