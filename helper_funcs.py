import time

def print_status_bar(epoch, stage, batch_i, total_batches, loss, t):
    
    bar_len = 24
    width = total_batches // bar_len
    batch_i += 1
    percentage = batch_i / total_batches * 100

    progress_done = ">" * (batch_i // width)
    progress_to_go = "." * (bar_len - (batch_i // width))
    
    print(f"\repoch: {epoch+1} \
        stage: {stage}\
        batch: {batch_i:03d}/{total_batches} \
        [{progress_done}>{progress_to_go}] ({percentage:.1f}%)\tloss: {loss:.5f}\tt:{(time.time() - t) // 60}min {(time.time() - t) % 60:.0f}s", end="")