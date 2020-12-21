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

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.logging.Logger;

/**
 * @author eric
 */
public class Utils {
  private final static Map<String, Character> GRID_MAP = Map.ofEntries(
      Map.entry("0000", ' '),
      Map.entry("0010", '▖'),
      Map.entry("0001", '▗'),
      Map.entry("1000", '▘'),
      Map.entry("0100", '▝'),
      Map.entry("1001", '▚'),
      Map.entry("0110", '▞'),
      Map.entry("1010", '▌'),
      Map.entry("0101", '▐'),
      Map.entry("0011", '▄'),
      Map.entry("1100", '▀'),
      Map.entry("1011", '▙'),
      Map.entry("0111", '▟'),
      Map.entry("1110", '▛'),
      Map.entry("1101", '▜'),
      Map.entry("1111", '█')
  );


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

  public static String grid(Grid<Boolean> grid, int l) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < l; i++) {
      Range<Integer> rx0 = Range.closedOpen(
          (int) Math.round((double) grid.getW() * (double) i / (double) l),
          (int) Math.round((double) grid.getW() * ((double) i + 1d / 2d) / (double) l)
      );
      Range<Integer> rx1 = Range.closedOpen(
          (int) Math.round((double) grid.getW() * ((double) i + 1d / 2d) / (double) l),
          (int) Math.round((double) grid.getW() * ((double) i + 1d) / (double) l)
      );
      Range<Integer> ry0 = Range.closedOpen(
          0,
          grid.getH() / 2
      );
      Range<Integer> ry1 = Range.closedOpen(
          grid.getH() / 2,
          grid.getH()
      );
      System.out.println(rx0);
      System.out.println(rx1);
      String binary = "" +
          (trueRatio(grid, rx0, ry1) >= 0.5d ? '1' : '0') +
          (trueRatio(grid, rx1, ry1) >= 0.5d ? '1' : '0') +
          (trueRatio(grid, rx0, ry0) >= 0.5d ? '1' : '0') +
          (trueRatio(grid, rx1, ry0) >= 0.5d ? '1' : '0');
      System.out.println(binary);
      sb.append(GRID_MAP.get(binary));
    }
    return sb.toString();
  }

  private static double trueRatio(Grid<Boolean> grid, Range<Integer> rangeX, Range<Integer> rangeY) {
    double nOfTrue = grid.stream().filter(e -> rangeX.contains(e.getX()) && rangeY.contains(e.getY()) && e.getValue()).count();
    double all = grid.stream().filter(e -> rangeX.contains(e.getX()) && rangeY.contains(e.getY())).count();
    return nOfTrue / all;
  }

  public static void main(String[] args) {
    Grid<Boolean> body = it.units.erallab.hmsrobots.util.Utils.buildShape("biped-4x4");
    System.out.println(Grid.toString(body, (Predicate<Boolean>) b -> b));
    System.out.println(grid(body, 2) + " " + grid(body, 4) + " " + grid(body, 8));
    body = it.units.erallab.hmsrobots.util.Utils.buildShape("tripod-7x3");
    System.out.println(grid(body, 2) + " " + grid(body, 4) + " " + grid(body, 8));
  }
}
