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

import com.google.common.collect.Range;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.util.TextPlotter;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * @author eric
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

  public static void main(String[] args) {
    Grid<Boolean> body = it.units.erallab.hmsrobots.util.Utils.buildShape("tripod-7x3");
    IntStream.range(2, 5).forEach(i -> System.out.println(TextPlotter.binaryMap(body.toArray(b->b), i)));
  }
}
