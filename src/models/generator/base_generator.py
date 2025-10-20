from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseGenerator(ABC):



    @abstractmethod
    def generate(self,
                 prompts: List[str],
                 system_instruction: str,
                 temperature: float = 0,
                 display_name: str = None,
                 in_batch=True,
                 structured_output: Optional[str] = None,
                 ) -> Dict[str, Any]:
        """
        return: {'status': str, responses: List[str]}
        """

        pass
