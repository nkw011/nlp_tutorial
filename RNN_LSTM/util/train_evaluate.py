from tqdm.notebook import tqdm

def train(model, optimizer, data, device):
    model.train()
    model.to(device)

    total_loss = 0.
    bar = tqdm(data, desc='train')

    batch_size = data.size()[1]
    hidden = model.init_hidden(batch_size)

    for i, x in enumerate(bar, start=1):
        x = x.to(device)

        if model.model_type == 'LSTM':
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)

        out, next_hidden = model(x, hidden) # out: (batch_size, seq_len, vocab_size)

        if model.model_type == 'LSTM':
            hidden = tuple(tensor.detach() for tensor in next_hidden) # detach()를 하지 않으면 backward()를 2번 한다는 RuntimeError가 발생한다.
        else:
            hidden = next_hidden.detach()

        # 다음 단어를 예측하는 것이므로 예측값에서 마지막 시점의 출력값은 제외하고, 정답에서는 2번째 시점부터 가져와 비교한다.
        # out을 tranpose하는 이유는 nll_loss가 input:(batch_size, num_class, dim1, dim2,...) target:(batch_size, dim1, dim2,) 방식으로 입력을 받기 때문이다.
        cost = F.nll_loss(out[:,:-1,:].transpose(1,2), x[:,1:])

        total_loss += cost.item()
        current_loss = total_loss / i

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


        bar.set_description(f"Train-loss:{current_loss:.4f}")


def evaluate(model, data, device, mode='valid'):
    model.eval()
    model.to(device)

    total_loss = 0.
    bar = tqdm(data, desc=mode)

    batch_size = data.size()[1]
    hidden = model.init_hidden(batch_size)

    loss_avg = 0.

    for i, x in enumerate(bar, start=1):
        with torch.no_grad():
            x = x.to(device)

            if model.model_type == 'LSTM':
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)

            out, next_hidden = model(x, hidden)  # out: (batch_size, seq_len, vocab_size)

            if model.model_type == 'LSTM':
                hidden = tuple(tensor.detach() for tensor in
                               next_hidden)  # detach()를 하지 않으면 backward()를 2번 한다는 RuntimeError가 발생한다.
            else:
                hidden = next_hidden.detach()

            # 다음 단어를 예측하는 것이므로 예측값에서 마지막 시점의 출력값은 제외하고, 정답에서는 2번째 시점부터 가져와 비교한다.
            # out을 tranpose하는 이유는 nll_loss가 input:(batch_size, num_class, dim1, dim2,...) target:(batch_size, dim1, dim2,) 방식으로 입력을 받기 때문이다.
            loss = F.nll_loss(out[:, :-1, :].transpose(1, 2), x[:, 1:])
        total_loss += loss.item()
        current_loss = total_loss / i
        loss_avg = current_loss

        bar.set_description(f"{mode}-loss:{current_loss:.4f}")

    return loss_avg