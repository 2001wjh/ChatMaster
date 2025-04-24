class AuthError(Exception):
    """自定义认证异常"""
    def __init__(self, message):
        self.message = message
        super().__init__(message)