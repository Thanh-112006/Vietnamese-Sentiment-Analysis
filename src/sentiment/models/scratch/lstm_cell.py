from torch import Tensor, nn


class CustomLSTMCell(nn.Module):
    """Cell LSTM tự viết với bốn cổng."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Khai báo tham số theo kích thước đầu vào và trạng thái ẩn."""
        raise NotImplementedError

    def forward(
        self, inputs: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Tính trạng thái ẩn và trạng thái ô nhớ của một bước."""
        raise NotImplementedError
