/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.logging.Logger;

/**
 * @author eric
 * @created 2020/08/19
 * @project VSREvolution
 */
public class Utils {

  private static final Logger L = Logger.getLogger(Utils.class.getName());

  private Utils() {
  }

  @SafeVarargs
  public static <K> List<K> concat(List<K>... lists) {
    List<K> all = new ArrayList<>();
    for (List<K> list : lists) {
      all.addAll(list);
    }
    return all;
  }

  public static <K, T> Function<K, T> ifThenElse(Predicate<K> predicate, Function<K, T> thenFunction, Function<K, T> elseFunction) {
    return k -> predicate.test(k) ? thenFunction.apply(k) : elseFunction.apply(k);
  }

  public static <K> SortedMap<Integer, K> index(List<K> list) {
    SortedMap<Integer, K> map = new TreeMap<>();
    for (int i = 0; i < list.size(); i++) {
      map.put(i, list.get(i));
    }
    return map;
  }
}
