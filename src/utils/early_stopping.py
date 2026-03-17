class EarlyStopping:
      """
        patience: 최고 점수와 비교할 때 몇 번 연속으로 개선이 없으면 멈출지 지정.
        min_delta: 최소 개선으로 인정하는 값.
        mode: 'min' or 'max'로 개선 방향 지정.
              min = 값이 낮아질수록 좋은 기준
              max = 값이 높아질수록 좋은 기준
      """
      def __init__(self, patience=5, min_delta=0, mode="min"):
          self.patience = patience
          self.min_delta = min_delta
          self.mode = mode
          self.best_score = None
          self.counter = 0
          self.stop = False

      def __call__(self, current_score):

          if self.stop:
             return

          if self.best_score is None:
              self.best_score = current_score
              return

          if self.mode == "min":
              improvement = current_score < self.best_score - self.min_delta
          elif self.mode == "max":
              improvement = current_score > self.best_score + self.min_delta
          else:
              raise ValueError("EarlyStopping mode should be 'min' or 'max'")

          if improvement:
              self.best_score = current_score
              self.counter = 0  # 개선 시 카운트 초기화
          else:
              self.counter += 1
              if self.counter >= self.patience:
                  self.stop = True