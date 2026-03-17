#CELL 9
model.eval()  # ONLY RUN THIS CELL IF TRAINED BEFORE, NOT WITH THE ALREADY TRAINED MODEL
#test loss
test_loss = evaluate_loss(model, test_loader, criterion)


print("-" * 30)
print(f"| Test Loss: {test_loss:.3f}")
print("-" * 30)


#CELL 10
model.eval()

def summarize_python_code(code_text, max_len=30):
    model.eval()

    #Preprocessing
    tokens = tokenizer_code(code_text)
    src_indexes = [vocab_code['<sos>']] + [vocab_code[token] for token in tokens] + [vocab_code['<eos>']]

    # Change unsequeeze to achieve batch_first=True
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    with torch.no_grad():

        encoder_outputs, hidden = model.encoder(src_tensor)

    # beggining point
    trg_indexes = [vocab_summary['<sos>']]

    # Greedy search
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            # Decoder predicts

            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        #We select the most probable
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)


        if pred_token == vocab_summary['<eos>']:
            break

    # Index to string
    itos = vocab_summary.get_itos()
    result_words = [itos[i] for i in trg_indexes
                    if i not in [vocab_summary['<sos>'], vocab_summary['<eos>'], vocab_summary['<pad>']]]

    return " ".join(result_words)

#Cualitative test
print("-" * 30)
print("TESTING GENERATION")
print("-" * 30)
test_examples = [
    "def factorial(n): return 1 if n == 0 else n * factorial(n - 1)",
    "def is_even(n): return n % 2 == 0",
    "def add(a, b): return a + b"
]

for ex in test_examples:
    print(f"Code: {ex}")
    print(f"Summary: {summarize_python_code(ex)}")
    print("-" * 10)