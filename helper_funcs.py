def print_status_bar(epoch, batch_i, total_batches, loss):
    
    bar_len = 24
    width = total_batches // bar_len
    percentage = batch_i / total_batches * 100

    progress_done = ">" * (batch_i // width)
    progress_to_go = "." * (bar_len - (batch_i // width))
    print(f"\repoch: {epoch+1}\t{batch_i:03d}/{total_batches}\t [{progress_done}>{progress_to_go}] ({percentage:.1f}%)\tloss: {loss:.5f} ", end="")




