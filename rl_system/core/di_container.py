"""
RL 시스템 의존성 주입 컨테이너
"""

from typing import Dict, Any, Optional, Type, TypeVar, Callable
from rl_system.config.rl_config import RLConfig
from rl_system.core.interfaces.data_provider import DataProvider
from rl_system.core.interfaces.model_interface import ModelInterface
from rl_system.core.interfaces.training_interface import TrainingInterface

T = TypeVar('T')


class RLContainer:
    """RL 시스템 의존성 주입 컨테이너"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._config: Optional[RLConfig] = None
        
        # 기본 서비스 등록
        self._register_default_services()
    
    def _register_default_services(self) -> None:
        """기본 서비스 등록"""
        # 설정 객체 등록
        self.register_singleton('config', lambda: RLConfig())
    
    def register_singleton(self, name: str, factory: Callable[[], T]) -> None:
        """
        싱글톤 서비스 등록
        
        Args:
            name: 서비스 이름
            factory: 서비스 생성 팩토리
        """
        self._factories[name] = factory
        if name in self._singletons:
            del self._singletons[name]
    
    def register_transient(self, name: str, factory: Callable[[], T]) -> None:
        """
        임시 서비스 등록 (매번 새로운 인스턴스 생성)
        
        Args:
            name: 서비스 이름
            factory: 서비스 생성 팩토리
        """
        self._services[name] = factory
    
    def register_instance(self, name: str, instance: T) -> None:
        """
        인스턴스 등록
        
        Args:
            name: 서비스 이름
            instance: 서비스 인스턴스
        """
        self._singletons[name] = instance
    
    def get(self, name: str) -> Any:
        """
        서비스 조회
        
        Args:
            name: 서비스 이름
            
        Returns:
            Any: 서비스 인스턴스
        """
        # 싱글톤 캐시에서 확인
        if name in self._singletons:
            return self._singletons[name]
        
        # 싱글톤 팩토리에서 생성
        if name in self._factories:
            instance = self._factories[name]()
            self._singletons[name] = instance
            return instance
        
        # 임시 서비스에서 생성
        if name in self._services:
            return self._services[name]()
        
        raise KeyError(f"Service '{name}' not found")
    
    def get_typed(self, name: str, service_type: Type[T]) -> T:
        """
        타입 지정 서비스 조회
        
        Args:
            name: 서비스 이름
            service_type: 서비스 타입
            
        Returns:
            T: 타입 지정된 서비스 인스턴스
        """
        instance = self.get(name)
        if not isinstance(instance, service_type):
            raise TypeError(f"Service '{name}' is not of type {service_type}")
        return instance
    
    def has(self, name: str) -> bool:
        """
        서비스 존재 여부 확인
        
        Args:
            name: 서비스 이름
            
        Returns:
            bool: 존재 여부
        """
        return (name in self._singletons or 
                name in self._factories or 
                name in self._services)
    
    def remove(self, name: str) -> None:
        """
        서비스 제거
        
        Args:
            name: 서비스 이름
        """
        if name in self._singletons:
            del self._singletons[name]
        if name in self._factories:
            del self._factories[name]
        if name in self._services:
            del self._services[name]
    
    def clear(self) -> None:
        """모든 서비스 제거"""
        self._singletons.clear()
        self._factories.clear()
        self._services.clear()
    
    def get_config(self) -> RLConfig:
        """
        설정 객체 조회
        
        Returns:
            RLConfig: 설정 객체
        """
        return self.get('config')
    
    def register_data_provider(self, name: str, provider: DataProvider) -> None:
        """
        데이터 제공자 등록
        
        Args:
            name: 제공자 이름
            provider: 데이터 제공자
        """
        self.register_instance(f"data_provider_{name}", provider)
    
    def get_data_provider(self, name: str) -> DataProvider:
        """
        데이터 제공자 조회
        
        Args:
            name: 제공자 이름
            
        Returns:
            DataProvider: 데이터 제공자
        """
        return self.get_typed(f"data_provider_{name}", DataProvider)
    
    def register_model(self, name: str, model: ModelInterface) -> None:
        """
        모델 등록
        
        Args:
            name: 모델 이름
            model: 모델 인스턴스
        """
        self.register_instance(f"model_{name}", model)
    
    def get_model(self, name: str) -> ModelInterface:
        """
        모델 조회
        
        Args:
            name: 모델 이름
            
        Returns:
            ModelInterface: 모델 인스턴스
        """
        return self.get_typed(f"model_{name}", ModelInterface)
    
    def register_trainer(self, name: str, trainer: TrainingInterface) -> None:
        """
        훈련자 등록
        
        Args:
            name: 훈련자 이름
            trainer: 훈련자 인스턴스
        """
        self.register_instance(f"trainer_{name}", trainer)
    
    def get_trainer(self, name: str) -> TrainingInterface:
        """
        훈련자 조회
        
        Args:
            name: 훈련자 이름
            
        Returns:
            TrainingInterface: 훈련자 인스턴스
        """
        return self.get_typed(f"trainer_{name}", TrainingInterface)
    
    def list_services(self) -> Dict[str, str]:
        """
        등록된 서비스 목록 조회
        
        Returns:
            Dict[str, str]: 서비스 이름과 타입
        """
        services = {}
        
        for name in self._singletons:
            services[name] = type(self._singletons[name]).__name__
        
        for name in self._factories:
            services[name] = "Factory"
        
        for name in self._services:
            services[name] = "Transient"
        
        return services
    
    def validate_configuration(self) -> List[str]:
        """
        설정 검증
        
        Returns:
            List[str]: 검증 오류 목록
        """
        errors = []
        
        # 필수 서비스 확인
        required_services = ['config']
        for service in required_services:
            if not self.has(service):
                errors.append(f"Required service '{service}' is not registered")
        
        # 설정 객체 검증
        try:
            config = self.get_config()
            if not isinstance(config, RLConfig):
                errors.append("Config service is not a valid RLConfig instance")
        except Exception as e:
            errors.append(f"Failed to get config: {e}")
        
        return errors


# 전역 컨테이너 인스턴스
_container = RLContainer()


def get_container() -> RLContainer:
    """전역 컨테이너 조회"""
    return _container


def reset_container() -> None:
    """전역 컨테이너 재설정"""
    global _container
    _container = RLContainer()