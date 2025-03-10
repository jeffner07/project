-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 06, 2024 at 04:49 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `driver_drowsiness`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `address` varchar(50) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `carno` varchar(20) NOT NULL,
  `rdate` varchar(15) NOT NULL,
  `owner` varchar(20) NOT NULL,
  `utype` varchar(20) NOT NULL,
  `fimg` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `mobile`, `email`, `address`, `uname`, `pass`, `carno`, `rdate`, `owner`, `utype`, `fimg`) VALUES
(1, 'Ram', 9894442716, 'ram@gmail.com', 'Salem', 'ram', '1234', 'TN2223', '14-04-2022', 'ram', 'owner', '1_40.jpg'),
(2, 'Rahul', 8890145671, 'rahul@gmail.com', 'Madurai', 'Rahul', '', 'TN2223', '14-04-2022', 'ram', 'user', '2_58.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `vt_alert`
--

CREATE TABLE `vt_alert` (
  `id` int(11) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `driver` varchar(20) NOT NULL,
  `message` varchar(50) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_alert`
--


-- --------------------------------------------------------

--
-- Table structure for table `vt_dd`
--

CREATE TABLE `vt_dd` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vimage` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_dd`
--

INSERT INTO `vt_dd` (`id`, `vid`, `vimage`) VALUES
(1, 1, '1_2.jpg'),
(2, 1, '1_3.jpg'),
(3, 1, '1_4.jpg'),
(4, 1, '1_5.jpg'),
(5, 1, '1_6.jpg'),
(6, 1, '1_7.jpg'),
(7, 1, '1_8.jpg'),
(8, 1, '1_9.jpg'),
(9, 1, '1_10.jpg'),
(10, 1, '1_11.jpg'),
(11, 1, '1_12.jpg'),
(12, 1, '1_13.jpg'),
(13, 1, '1_14.jpg'),
(14, 1, '1_15.jpg'),
(15, 1, '1_16.jpg'),
(16, 1, '1_17.jpg'),
(17, 1, '1_18.jpg'),
(18, 1, '1_19.jpg'),
(19, 1, '1_20.jpg'),
(20, 1, '1_21.jpg'),
(21, 1, '1_22.jpg'),
(22, 1, '1_23.jpg'),
(23, 1, '1_24.jpg'),
(24, 1, '1_25.jpg'),
(25, 1, '1_26.jpg'),
(26, 1, '1_27.jpg'),
(27, 1, '1_28.jpg'),
(28, 1, '1_29.jpg'),
(29, 1, '1_30.jpg'),
(30, 1, '1_31.jpg'),
(31, 1, '1_32.jpg'),
(32, 1, '1_33.jpg'),
(33, 1, '1_34.jpg'),
(34, 1, '1_35.jpg'),
(35, 1, '1_36.jpg'),
(36, 1, '1_37.jpg'),
(37, 1, '1_38.jpg'),
(38, 1, '1_39.jpg'),
(39, 1, '1_40.jpg'),
(40, 2, '2_2.jpg'),
(41, 2, '2_3.jpg'),
(42, 2, '2_4.jpg'),
(43, 2, '2_5.jpg'),
(44, 2, '2_6.jpg'),
(45, 2, '2_7.jpg'),
(46, 2, '2_8.jpg'),
(47, 2, '2_9.jpg'),
(48, 2, '2_10.jpg'),
(49, 2, '2_11.jpg'),
(50, 2, '2_12.jpg'),
(51, 2, '2_13.jpg'),
(52, 2, '2_14.jpg'),
(53, 2, '2_15.jpg'),
(54, 2, '2_16.jpg'),
(55, 2, '2_17.jpg'),
(56, 2, '2_18.jpg'),
(57, 2, '2_19.jpg'),
(58, 2, '2_20.jpg'),
(59, 2, '2_21.jpg'),
(60, 2, '2_22.jpg'),
(61, 2, '2_23.jpg'),
(62, 2, '2_24.jpg'),
(63, 2, '2_25.jpg'),
(64, 2, '2_26.jpg'),
(65, 2, '2_27.jpg'),
(66, 2, '2_28.jpg'),
(67, 2, '2_29.jpg'),
(68, 2, '2_30.jpg'),
(69, 2, '2_31.jpg'),
(70, 2, '2_32.jpg'),
(71, 2, '2_33.jpg'),
(72, 2, '2_34.jpg'),
(73, 2, '2_35.jpg'),
(74, 2, '2_36.jpg'),
(75, 2, '2_37.jpg'),
(76, 2, '2_38.jpg'),
(77, 2, '2_39.jpg'),
(78, 2, '2_40.jpg'),
(79, 3, '3_2.jpg'),
(80, 3, '3_3.jpg'),
(81, 3, '3_4.jpg'),
(82, 3, '3_5.jpg'),
(83, 3, '3_6.jpg'),
(84, 3, '3_7.jpg'),
(85, 3, '3_8.jpg'),
(86, 3, '3_9.jpg'),
(87, 3, '3_10.jpg'),
(88, 3, '3_11.jpg'),
(89, 3, '3_12.jpg'),
(90, 3, '3_13.jpg'),
(91, 3, '3_14.jpg'),
(92, 3, '3_15.jpg'),
(93, 3, '3_16.jpg'),
(94, 3, '3_17.jpg'),
(95, 3, '3_18.jpg'),
(96, 3, '3_19.jpg'),
(97, 3, '3_20.jpg'),
(98, 3, '3_21.jpg'),
(99, 3, '3_22.jpg'),
(100, 3, '3_23.jpg'),
(101, 3, '3_24.jpg'),
(102, 3, '3_25.jpg'),
(103, 3, '3_26.jpg'),
(104, 3, '3_27.jpg'),
(105, 3, '3_28.jpg'),
(106, 3, '3_29.jpg'),
(107, 3, '3_30.jpg'),
(108, 3, '3_31.jpg'),
(109, 3, '3_32.jpg'),
(110, 3, '3_33.jpg'),
(111, 3, '3_34.jpg'),
(112, 3, '3_35.jpg'),
(113, 3, '3_36.jpg'),
(114, 3, '3_37.jpg'),
(115, 3, '3_38.jpg'),
(116, 3, '3_39.jpg'),
(117, 3, '3_40.jpg'),
(118, 4, '4_2.jpg'),
(119, 4, '4_3.jpg'),
(120, 4, '4_4.jpg'),
(121, 4, '4_5.jpg'),
(122, 4, '4_6.jpg'),
(123, 4, '4_7.jpg'),
(124, 4, '4_8.jpg'),
(125, 4, '4_9.jpg'),
(126, 4, '4_10.jpg'),
(127, 4, '4_11.jpg'),
(128, 4, '4_12.jpg'),
(129, 4, '4_13.jpg'),
(130, 4, '4_14.jpg'),
(131, 4, '4_15.jpg'),
(132, 4, '4_16.jpg'),
(133, 4, '4_17.jpg'),
(134, 4, '4_18.jpg'),
(135, 4, '4_19.jpg'),
(136, 4, '4_20.jpg'),
(137, 4, '4_21.jpg'),
(138, 4, '4_22.jpg'),
(139, 4, '4_23.jpg'),
(140, 4, '4_24.jpg'),
(141, 4, '4_25.jpg'),
(142, 4, '4_26.jpg'),
(143, 4, '4_27.jpg'),
(144, 4, '4_28.jpg'),
(145, 4, '4_29.jpg'),
(146, 4, '4_30.jpg'),
(147, 4, '4_31.jpg'),
(148, 4, '4_32.jpg'),
(149, 4, '4_33.jpg'),
(150, 4, '4_34.jpg'),
(151, 4, '4_35.jpg'),
(152, 4, '4_36.jpg'),
(153, 4, '4_37.jpg'),
(154, 4, '4_38.jpg'),
(155, 4, '4_39.jpg'),
(156, 4, '4_40.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `vt_face`
--

CREATE TABLE `vt_face` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vface` varchar(20) NOT NULL,
  `status` int(11) NOT NULL,
  `owner` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_face`
--

INSERT INTO `vt_face` (`id`, `vid`, `vface`, `status`, `owner`) VALUES
(40, 3, '3_2.jpg', 0, 'vishnu'),
(41, 3, '3_3.jpg', 0, 'vishnu'),
(42, 3, '3_4.jpg', 0, 'vishnu'),
(43, 3, '3_5.jpg', 0, 'vishnu'),
(44, 3, '3_6.jpg', 0, 'vishnu'),
(45, 3, '3_7.jpg', 0, 'vishnu'),
(46, 3, '3_8.jpg', 0, 'vishnu'),
(47, 3, '3_9.jpg', 0, 'vishnu'),
(48, 3, '3_10.jpg', 0, 'vishnu'),
(49, 3, '3_11.jpg', 0, 'vishnu'),
(50, 3, '3_12.jpg', 0, 'vishnu'),
(51, 3, '3_13.jpg', 0, 'vishnu'),
(52, 3, '3_14.jpg', 0, 'vishnu'),
(53, 3, '3_15.jpg', 0, 'vishnu'),
(54, 3, '3_16.jpg', 0, 'vishnu'),
(55, 3, '3_17.jpg', 0, 'vishnu'),
(56, 3, '3_18.jpg', 0, 'vishnu'),
(57, 3, '3_19.jpg', 0, 'vishnu'),
(58, 3, '3_20.jpg', 0, 'vishnu'),
(59, 3, '3_21.jpg', 0, 'vishnu'),
(60, 3, '3_22.jpg', 0, 'vishnu'),
(61, 3, '3_23.jpg', 0, 'vishnu'),
(62, 3, '3_24.jpg', 0, 'vishnu'),
(63, 3, '3_25.jpg', 0, 'vishnu'),
(64, 3, '3_26.jpg', 0, 'vishnu'),
(65, 3, '3_27.jpg', 0, 'vishnu'),
(66, 3, '3_28.jpg', 0, 'vishnu'),
(67, 3, '3_29.jpg', 0, 'vishnu'),
(68, 3, '3_30.jpg', 0, 'vishnu'),
(69, 3, '3_31.jpg', 0, 'vishnu'),
(70, 3, '3_32.jpg', 0, 'vishnu'),
(71, 3, '3_33.jpg', 0, 'vishnu'),
(72, 3, '3_34.jpg', 0, 'vishnu'),
(73, 3, '3_35.jpg', 0, 'vishnu'),
(74, 3, '3_36.jpg', 0, 'vishnu'),
(75, 3, '3_37.jpg', 0, 'vishnu'),
(76, 3, '3_38.jpg', 0, 'vishnu'),
(77, 3, '3_39.jpg', 0, 'vishnu'),
(78, 3, '3_40.jpg', 0, 'vishnu'),
(79, 4, '4_2.jpg', 0, 'raj'),
(80, 4, '4_3.jpg', 0, 'raj'),
(81, 4, '4_4.jpg', 0, 'raj'),
(82, 4, '4_5.jpg', 0, 'raj'),
(83, 4, '4_6.jpg', 0, 'raj'),
(84, 4, '4_7.jpg', 0, 'raj'),
(85, 4, '4_8.jpg', 0, 'raj'),
(86, 4, '4_9.jpg', 0, 'raj'),
(87, 4, '4_10.jpg', 0, 'raj'),
(88, 4, '4_11.jpg', 0, 'raj'),
(89, 4, '4_12.jpg', 0, 'raj'),
(90, 4, '4_13.jpg', 0, 'raj'),
(91, 4, '4_14.jpg', 0, 'raj'),
(92, 4, '4_15.jpg', 0, 'raj'),
(93, 4, '4_16.jpg', 0, 'raj'),
(94, 4, '4_17.jpg', 0, 'raj'),
(95, 4, '4_18.jpg', 0, 'raj'),
(96, 4, '4_19.jpg', 0, 'raj'),
(97, 4, '4_20.jpg', 0, 'raj'),
(98, 4, '4_21.jpg', 0, 'raj'),
(99, 4, '4_22.jpg', 0, 'raj'),
(100, 4, '4_23.jpg', 0, 'raj'),
(101, 4, '4_24.jpg', 0, 'raj'),
(102, 4, '4_25.jpg', 0, 'raj'),
(103, 4, '4_26.jpg', 0, 'raj'),
(104, 4, '4_27.jpg', 0, 'raj'),
(105, 4, '4_28.jpg', 0, 'raj'),
(106, 4, '4_29.jpg', 0, 'raj'),
(107, 4, '4_30.jpg', 0, 'raj'),
(108, 4, '4_31.jpg', 0, 'raj'),
(109, 4, '4_32.jpg', 0, 'raj'),
(110, 4, '4_33.jpg', 0, 'raj'),
(111, 4, '4_34.jpg', 0, 'raj'),
(112, 4, '4_35.jpg', 0, 'raj'),
(113, 4, '4_36.jpg', 0, 'raj'),
(114, 4, '4_37.jpg', 0, 'raj'),
(115, 4, '4_38.jpg', 0, 'raj'),
(116, 4, '4_39.jpg', 0, 'raj'),
(117, 4, '4_40.jpg', 0, 'raj'),
(118, 1, '1_2.jpg', 0, 'ram'),
(119, 1, '1_3.jpg', 0, 'ram'),
(120, 1, '1_4.jpg', 0, 'ram'),
(121, 1, '1_5.jpg', 0, 'ram'),
(122, 1, '1_6.jpg', 0, 'ram'),
(123, 1, '1_7.jpg', 0, 'ram'),
(124, 1, '1_8.jpg', 0, 'ram'),
(125, 1, '1_9.jpg', 0, 'ram'),
(126, 1, '1_10.jpg', 0, 'ram'),
(127, 1, '1_11.jpg', 0, 'ram'),
(128, 1, '1_12.jpg', 0, 'ram'),
(129, 1, '1_13.jpg', 0, 'ram'),
(130, 1, '1_14.jpg', 0, 'ram'),
(131, 1, '1_15.jpg', 0, 'ram'),
(132, 1, '1_16.jpg', 0, 'ram'),
(133, 1, '1_17.jpg', 0, 'ram'),
(134, 1, '1_18.jpg', 0, 'ram'),
(135, 1, '1_19.jpg', 0, 'ram'),
(136, 1, '1_20.jpg', 0, 'ram'),
(137, 1, '1_21.jpg', 0, 'ram'),
(138, 1, '1_22.jpg', 0, 'ram'),
(139, 1, '1_23.jpg', 0, 'ram'),
(140, 1, '1_24.jpg', 0, 'ram'),
(141, 1, '1_25.jpg', 0, 'ram'),
(142, 1, '1_26.jpg', 0, 'ram'),
(143, 1, '1_27.jpg', 0, 'ram'),
(144, 1, '1_28.jpg', 0, 'ram'),
(145, 1, '1_29.jpg', 0, 'ram'),
(146, 1, '1_30.jpg', 0, 'ram'),
(147, 1, '1_31.jpg', 0, 'ram'),
(148, 1, '1_32.jpg', 0, 'ram'),
(149, 1, '1_33.jpg', 0, 'ram'),
(150, 1, '1_34.jpg', 0, 'ram'),
(151, 1, '1_35.jpg', 0, 'ram'),
(152, 1, '1_36.jpg', 0, 'ram'),
(153, 1, '1_37.jpg', 0, 'ram'),
(154, 1, '1_38.jpg', 0, 'ram'),
(155, 1, '1_39.jpg', 0, 'ram'),
(156, 1, '1_40.jpg', 0, 'ram');
