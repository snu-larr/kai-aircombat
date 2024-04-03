
import torch
import torch.nn as nn


class GRULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        
        """GRULayer 클래스의 생성자.

        GRU 기반의 순환 신경망 레이어를 초기화한다. 이 레이어는 시퀀스 데이터 처리에 적합하며,
        주어진 입력 크기, 숨겨진 상태의 크기, 레이어의 개수를 기반으로 구성된다.

        매개변수:
            input_size (int): 각 입력 시퀀스의 특성의 수.
            hidden_size (int): 숨겨진 상태의 크기.
            num_layers (int): GRU 레이어의 개수.

        속성:
            gru (nn.GRU): PyTorch의 GRU 모듈. 시퀀스 데이터를 처리한다.
            norm (nn.LayerNorm): 숨겨진 상태에 대한 정규화를 수행하는 LayerNorm.

        사용 예시:
            >>> gru_layer = GRULayer(input_size=10, hidden_size=20, num_layers=2)
            >>> print(gru_layer)
            GRULayer(
                (gru): GRU(10, 20, num_layers=2)
                (norm): LayerNorm((20,), eps=1e-05, elementwise_affine=True)
            )
        이 클래스는 시퀀스 데이터를 효과적으로 처리하기 위해 GRU 레이어와 숨겨진 상태의 정규화 기능을 제공한다.
        """
        
        super(GRULayer, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
        # NOTE: self.gru(x, hxs) needs x=[T, N, input_size] and hxs=[L, N, hidden_size]

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor):

        """GRU 레이어의 순방향 패스를 실행한다.

        이 메소드는 입력 데이터 `x`, 이전 숨겨진 상태 `hxs`, 그리고 마스크 `masks`를 받아
        GRU 네트워크를 통해 다음 숨겨진 상태를 계산하고, 결과적으로 순방향 패스의 출력을 반환한다.

        매개변수:
            x (torch.Tensor): 입력 데이터. [배치 크기, 입력 크기] 또는 [시퀀스 길이 * 배치 크기, 입력 크기]의 형태.
            hxs (torch.Tensor): 이전 숨겨진 상태. [배치 크기, 숨겨진 레이어 수, 숨겨진 크기].
            masks (torch.Tensor): 입력 데이터의 마스크. [배치 크기, 1] 또는 [시퀀스 길이 * 배치 크기, 1].

        반환값:
            torch.Tensor: GRU 레이어의 출력.
            torch.Tensor: 다음 숨겨진 상태.

        사용 예시:
            >>> gru_layer = GRULayer(input_size=10, hidden_size=20, num_layers=2)
            >>> x = torch.randn(5, 10)  # 5는 배치 크기, 10은 입력 크기
            >>> hxs = torch.zeros(5, 2, 20)  # 5는 배치 크기, 2는 숨겨진 레이어 수, 20은 숨겨진 크기
            >>> masks = torch.ones(5, 1)  # 시퀀스 내 모든 요소를 고려함
            >>> output, nhxs = gru_layer(x, hxs, masks)
        """
        
        # NOTE: N = mini_batch_size; T = recurrent chunk length; L = gru layers

        # (T=1) x: [N, input_size], hxs: [N, L, hidden_size], masks: [N, 1]
        if x.size(0) == hxs.size(0):
            # masks: [N, 1] => [N, L] => [N, L, 1]
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks.repeat(1, self._num_layers).unsqueeze(-1)).transpose(0, 1).contiguous())

            x = x.squeeze(0)            # [1, N, input_size] => [N, input_size]
            hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]

        # (T>1): x=[T * N, input_size], hxs=[N, L, hidden_size], masks=[T * N, 1]
        else:
            # Mannual reset hxs to zero at ternimal states might be too slow to calculate
            # We need to tackle the problem more efficiently

            # x is a (T, N, input_size) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)
            # unflatten x and masks
            x = x.view(T, N, x.size(1))  # [T * N, input_size] => [T, N, input_size]
            masks = masks.view(T, N)     # [T * N, 1] => [T, N]

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)       # [T, N] => [T, 1]
                         .nonzero(as_tuple=False)
                         .squeeze(dim=-1)   # [T, 1] => [T]
                         .cpu())
            # +1 to correct the masks[1:]
            has_zeros = (has_zeros + 1).numpy().tolist()
            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)   # [N, L, hidden_size] => [L, N, hidden_size]
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                # masks[start_idx]: [N] => [1, N, 1] => [L, N, 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._num_layers, 1, 1)).contiguous()
                rnn_scores, hxs = self.gru(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.view(T * N, -1)       # [T, N, input_size] => [T * N, input_size]
            hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]

        x = self.norm(x)
        return x, hxs

    @property
    def output_size(self):
        
        """GRU 레이어의 출력 크기를 반환한다.

        이 속성은 GRU 레이어를 통과한 데이터의 숨겨진 상태 벡터의 크기, 즉 숨겨진 크기를 나타낸다.
        이는 레이어의 구성 파라미터 중 하나로 설정되며, 레이어의 출력 차원을 결정한다.

        반환값:
            int: GRU 레이어의 출력 크기, 즉 숨겨진 상태 벡터의 크기.

        사용 예시:
            >>> gru_layer = GRULayer(input_size=10, hidden_size=20, num_layers=2)
            >>> print(gru_layer.output_size)
            20
        """
        
        return self._hidden_size
