import numpy as np
from collections import Counter

# ах да, еще мне нравятся плюсы и типизация, поэтому...
from typing import List, Dict, Any, Optional, Union, Tuple


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    sort_order: np.ndarray = np.argsort(feature_vector)
    sorted_features: np.ndarray = feature_vector[sort_order]
    sorted_targets: np.ndarray = target_vector[sort_order]
    n_samples: int = len(feature_vector)

    # в этих точках значения меняются
    change_points: np.ndarray = np.where(sorted_features[:-1] != sorted_features[1:])[0]

    if change_points.size == 0:
        return np.array([]), np.array([]), None, None

    thresholds: np.ndarray = (
        sorted_features[change_points] + sorted_features[change_points + 1]
    ) / 2

    # Кумулятивная сумма — это нарастающий итог значений, вычисляемый путем последовательного сложения всех предыдущих значений в ряду (а я верю google)
    cum_class_0: np.ndarray = np.cumsum(sorted_targets == 0)
    cum_class_1: np.ndarray = np.cumsum(sorted_targets == 1)
    total_class_1: int = int(cum_class_1[-1])
    total_class_0: int = int(cum_class_0[-1])

    left_size: np.ndarray = change_points + 1
    right_size: np.ndarray = n_samples - left_size

    # Количество классов в левом поддереве
    left_class_1: np.ndarray = cum_class_1[change_points]
    left_class_0: np.ndarray = cum_class_0[change_points]

    # Количество классов в правом поддереве
    right_class_1: np.ndarray = total_class_1 - left_class_1
    right_class_0: np.ndarray = total_class_0 - left_class_0

    # Вероятноти классов в поддеревьяв по обычной формуле P = n / N
    class_1_left: np.ndarray = left_class_1 / left_size
    class_0_left: np.ndarray = left_class_0 / left_size

    class_1_right: np.ndarray = right_class_1 / right_size
    class_0_right: np.ndarray = right_class_0 / right_size

    # Вычисляем энтропию Джини для поддеревьев
    gini_left: np.ndarray = 1 - class_1_left**2 - class_0_left**2
    gini_right: np.ndarray = 1 - class_1_right**2 - class_0_right**2

    # Вычисляем общий критерий Джини для каждого порога
    gini_scores: np.ndarray = (
        -(left_size / n_samples) * gini_left - (right_size / n_samples) * gini_right
    )

    # Находим лучший порог
    best_index = np.argmax(gini_scores)

    return thresholds, gini_scores, thresholds[best_index], gini_scores[best_index]


import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple


class DecisionTree:
    def __init__(
        self,
        feature_types: List[str],
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
    ) -> None:
        if np.any(
            list(map(lambda x: x != "real" and x != "categorical", feature_types))
        ):
            raise ValueError("There is unknown feature type")

        self._tree: Dict[str, Any] = {}
        self._feature_types: List[str] = feature_types
        self._max_depth: Optional[int] = max_depth
        self._min_samples_split: Optional[int] = min_samples_split
        self._min_samples_leaf: Optional[int] = min_samples_leaf

    def _fit_node(
        self, sub_X: np.ndarray, sub_y: np.ndarray, node: Dict[str, Any], cur_depth: int
    ) -> None:
        if np.all(sub_y == sub_y[0]):  # а вот туть было !=
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._max_depth is not None and cur_depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best: Optional[int] = None
        threshold_best: Optional[Union[float, List[str]]] = None
        gini_best: Optional[float] = None
        split: Optional[np.ndarray] = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map: Dict[str, int] = {}

            if feature_type == "real":
                feature_vector: np.ndarray = sub_X[:, feature]
            elif feature_type == "categorical":
                counts: Counter = Counter(sub_X[:, feature])
                clicks: Counter = Counter(sub_X[sub_y == 1, feature])
                ratio: Dict[str, float] = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = (
                        0
                        if current_click == 0
                        else current_count
                        / current_click  # а тут теперь нет деления на 0
                    )
                sorted_categories: List[str] = [
                    x[0] for x in sorted(ratio.items(), key=lambda x: x[1])
                ]
                categories_map = dict(
                    zip(sorted_categories, range(len(sorted_categories)))
                )
                feature_vector = np.array(
                    [categories_map[x] for x in sub_X[:, feature]]
                )
            else:
                raise ValueError(f"не понимяу фичу: {feature_type}")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None:
                continue

            if gini_best is None or (gini is not None and gini > gini_best):
                left: np.ndarray = feature_vector < threshold
                right: np.ndarray = ~left

                if self._min_samples_leaf is not None and (
                    left.sum() < self._min_samples_leaf
                    and right.sum() < self._min_samples_leaf
                ):
                    continue

                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [
                        x[0]
                        for x in filter(
                            lambda x: x[1] < threshold, categories_map.items()  # type: ignore
                        )
                    ]
                else:
                    raise ValueError(f"Нет такой фичи: {feature_type}")

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError(f"Нет такой фичи: {self._feature_types[feature_best]}")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], cur_depth + 1)
        self._fit_node(
            sub_X[np.logical_not(split)],  # type: ignore
            sub_y[np.logical_not(split)],  # type: ignore
            node["right_child"],
            cur_depth + 1,
        )

    def _predict_node(self, x: np.ndarray, node: Dict[str, Any]) -> Any:
        if node["type"] == "terminal":
            return node["class"]

        feature_ix: int = node["feature_split"]
        feature_val = x[feature_ix]

        if self._feature_types[feature_ix] == "real":
            if feature_val < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_ix] == "categorical":
            if feature_val in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Нет такой фичи")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._tree = {}
        self._fit_node(X, y, self._tree, cur_depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predicted: List[Any] = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params: Any) -> "DecisionTree":
        if "feature_types" in params:
            self._feature_types = params["feature_types"]
        if "max_depth" in params:
            self._max_depth = params["max_depth"]
        if "min_samples_split" in params:
            self._min_samples_split = params["min_samples_split"]
        if "min_samples_leaf" in params:
            self._min_samples_leaf = params["min_samples_leaf"]
        return self
