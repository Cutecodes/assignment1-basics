import torch
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'epoch': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint["epoch"]

class IndexedHeap:
    """支持修改任意元素的堆, 小顶堆"""
    
    def __init__(self, is_min_heap: bool = True):
        self.heap = []  # (value, key)
        self.index_map = {}  # key -> index in heap
        self.is_min_heap = is_min_heap
    
    def push(self, key, value):
        if key in self.index_map:
            raise KeyError(f"key '{key}' already exists")

        value = value.copy()
        self.heap.append((value, key))
        self.index_map[key] = len(self.heap) - 1

        self._sift_up(len(self.heap) - 1)
    
    def update(self, key, new_value):

        if key not in self.index_map:
            self.push(key, new_value)
            return
        
        
        idx = self.index_map[key]
        old_value = self.heap[idx][0]
        
        # update value
        new_value = new_value.copy()
        self.heap[idx] = (new_value, key)
        
        # rebuild heap
        if self.is_min_heap:
            need_up = new_value < old_value
        else:
            need_up = new_value > old_value
        
        if need_up:
            self._sift_up(idx)
        else:
            self._sift_down(idx)
    
    def delete(self, key):
        """delete key(hard delete)"""
        if key not in self.index_map:
            return
        
        # get and delete index
        idx = self.index_map.pop(key)
        
        # delete heap
        last_idx = len(self.heap) - 1
        if idx == last_idx:
            self.heap.pop()
        else:
            # swap and update index
            self.heap[idx], self.heap[last_idx] = self.heap[last_idx], self.heap[idx]
            self.index_map[self.heap[idx][1]] = idx
            self.heap.pop()
            # adjust heap
            self._sift_up(idx)
            self._sift_down(idx)
    
    def peek(self):
        """peek heap"""
        if self.heap:
            return self.heap[0][1], self.heap[0][0]
        return None, None

    def pop(self):
        """pop heap"""
        if not self.heap:
            return None, None
        
        min_val, min_key = self.heap[0]
        self.delete(min_key)
        return min_key, min_val
    
    def _sift_up(self, i):
        heap = self.heap
        while i > 0:
            parent = (i - 1) // 2
            if self.is_min_heap:
                need_swap = heap[i][0] < heap[parent][0]
            else:
                need_swap = heap[i][0] > heap[parent][0]

            if need_swap:
                heap[i], heap[parent] = heap[parent], heap[i]
                # update index
                self.index_map[heap[i][1]] = i
                self.index_map[heap[parent][1]] = parent
                i = parent
            else:
                break
    
    def _sift_down(self, i):
        heap = self.heap
        size = len(heap)
        
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if self.is_min_heap:
                if left < size and heap[left][0] < heap[smallest][0]:
                    smallest = left
                if right < size and heap[right][0] < heap[smallest][0]:
                    smallest = right
            else:
                if left < size and heap[left][0] > heap[smallest][0]:
                    smallest = left
                if right < size and heap[right][0] > heap[smallest][0]:
                    smallest = right
            
            if smallest == i:
                break
            
            heap[i], heap[smallest] = heap[smallest], heap[i]
            self.index_map[heap[i][1]] = i
            self.index_map[heap[smallest][1]] = smallest
            i = smallest
